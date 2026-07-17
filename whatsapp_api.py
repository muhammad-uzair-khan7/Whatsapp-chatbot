from pywa import WhatsApp, filters, types
from pywa.types import MessageType
from dotenv import load_dotenv
import os
import base64
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated, Sequence

from fastapi import FastAPI, Request


#Credentials import from .env file
load_dotenv()
PHONENUMBER_ID= os.getenv("PHONENUMBER_ID") #PHONENUMBER_ID is the WhatsApp Phone ID from Meta App Dashboard.
ACCESS_TOKEN= os.getenv("ACCESS_TOKEN") #ACCES_TOKEN is the WhatsApp App Access Token from Meta App Dasboard
APP_ID= os.getenv("APP_ID") #APP_ID is the whatsapp app id from Facebook apps
APP_SECRET=os.getenv("APP_SECRET") #APP_SECRET is the wahtsapp app secret from Facebook apps
VERIFY_TOKEN= os.getenv("VERIFY_TOKEN") #This is the verify token you set in your webhook configuration in the Meta App Dashboard. It is used to verify that incoming requests to your webhook are from WhatsApp.
GOOGLE_GENERATIVE_AI= os.getenv("GEMINI_API_KEY") #This is Gemini API key
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
POS_BASE_URL= os.getenv("POS_BASE_URL")

openrouter_model= OpenAI(api_key=OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")
app= FastAPI(title="WhatsApp API Server")
wa = WhatsApp(phone_id=PHONENUMBER_ID, token=ACCESS_TOKEN, app_id=APP_ID, app_secret=APP_SECRET, callback_url="https://overvaliant-waneta-optometrical.ngrok-free.dev", server=app, webhook_endpoint="/whatsapp/webhook", verify_token=VERIFY_TOKEN)

def get_chat_model():
    chat_model= ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, api_key=GOOGLE_GENERATIVE_AI)
    return chat_model

# State Class For ChatBot    
class BotState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    incoming_message: str
    wa_id: int
    classifier: Literal["order_management", "customer_query", "customer_complaint"]
    response_to_user: str

def extract_text(ai_message):
    content = ai_message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def extract_text_message(message: types.Message):
    if message.type == MessageType.TEXT:
        return message.text
    elif message.type== MessageType.AUDIO:
        if message.audio.voice:
            print("Downloading voice note...")
            audio_bytes= message.audio.get_bytes()
            base64audio= base64.b64encode(audio_bytes).decode('utf-8')
            try:
                response= openrouter_model.chat.completions.create(model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                                                                   messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe this audio precisely. Output ONLY the transcription text, nothing else."},
                            # Pass audio context using OpenRouter's multimodal format
                            {"type": "input_audio", "input_audio": {"data": base64audio, "format": "ogg"}}
                        ]
                    }
                ])
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenRouter error in STT: {e}")

    elif message.type== MessageType.IMAGE:
        image_bytes= message.image.get_bytes()
        base64_image= base64.b64encode(image_bytes).decode('utd-8')
        caption= message.image.caption or ""
        try:
            response = openrouter_model.chat.completions.create(
                model="nvidia/nemotron-nano-12b-2-vl:free",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text from this image or describe the image clearly so a search engine can index it. If it's not text, analzye the iamge and tell describe what it is."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
            )
            extracted_visual_text= response.choices[0].message.content
            return f"User Caption: {caption} | Image content : {extracted_visual_text}".strip()
        except Exception as e:
            print(f"Openrouter Image processing error: {e}")
    return ""

class IncomingMessageParser(BaseModel):
    #This will act as a specific output parser/format for the response of AI model.
    category: Literal["order_management", "customer_query", "customer_complaint"]= Field(description="The category of incoming message.")


def classifier_prompt():
    prompt_template= ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that classifies incoming WhatsApp messages into one of following categories: order_management, customer_query, customer_complaint. You should only respond with the category name, and nothing else."),
            ("human", "Classify the following message: {message}")
        ]
    )
    return prompt_template

def structured_llm_model(chat_model: ChatGoogleGenerativeAI= get_chat_model()):
    return chat_model.with_structured_output(IncomingMessageParser)
#----------------------Agent 1----------------------

def classifier_agent(state: BotState, chat_model: ChatGoogleGenerativeAI= get_chat_model()):
    #1. Receive the incoming message from the user.
    message= state["incoming_message"]
    #2. Use an AI Agent to classify the message into one of the predefined categories (e.g, order management, customer query, customer complaint, feedback collection).
    prompt= classifier_prompt()
    structured_model= structured_llm_model()
    chain= prompt | structured_model
    response= chain.invoke({"message": message})
    return {"classifier": response.category}
    #3. Based on classification, route the message to the appropriate agent for further processing.
    """This step will be executed in next functions where we will define the agents for each category and route the message accordingly."""

#-----------------RAG Tool ------------------------
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

_VECTORSTORE_PATH= "vectorstore"
_embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_or_load_vectorstore():
    """
    Builds the FAISS index from cafe_details.txt if it doesn't exist yet,
    otherwise loads the cached index from disk. Avoids re-embedding on
    every single message.
    """
    if os.path.exists(_VECTORSTORE_PATH):
        return FAISS.load_local(
            _VECTORSTORE_PATH, _embeddings, allow_dangerous_deserialization=True
        )
    loader= TextLoader("./cafe_details.txt")
    docs= loader.load()
    #Process the document and split it into chunks
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    chunks= text_splitter.split_documents(docs)
    vectorstore= FAISS.from_documents(chunks, _embeddings)
    vectorstore.save_local("_VECTORSTORE_PATH")
    return vectorstore

_vectorstore= build_or_load_vectorstore()

def Rag_tool():
    return _vectorstore.as_retriever(search_kwargs={"k":3})

#----------------------Agent 2------------------------
def customer_query_agent(state:BotState):
    chat_model=get_chat_model()
    retriever= Rag_tool()
    prompts= ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert customer service representative and you'll answer all customer queries very respectfully. Always look in the context/details of cafe first, then answer to customer. "
            "\nContext: {context}\n"
            "If customer sends an greeting message, greet him back respectfully."
            "But be precise and short with your answers."
            "You may also receive messages in roman english, that would be sent by Pakistanis, so accomodate such customers with their way."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )
    #Use the retreiver to fetch relevant information from the document
    question_answer_chain= create_stuff_documents_chain(llm=chat_model, prompt=prompts)
    rag_chain= create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)
    history= state.get("messages", [])
    current_user_message= state["incoming_message"]
    response= rag_chain.invoke({"chat_history": history, "input": current_user_message})
    answer= response["answer"]
    return {
        "messages": [AIMessage(content=answer)],
        "response_to_user": answer
    }

#-------------Agent 2-----------------
#-----------Tools--------------
from order_management import create_order, check_order_status, see_item_stock
model= get_chat_model()
_order_tools = [create_order, check_order_status, see_item_stock]
order_tool_based_model= model.bind_tools(tools=_order_tools)
def order_management_agent(state: BotState):
    prompt= ChatPromptTemplate.from_messages(
        [('system', """You are a cafe receptionist who create orders for customer over online messages and forwards order status with customers.
        # GUIDELINES FOR ORDER CREATION:
        * First, get the item name, with quantities from user.
        * Once you get the item name with quantities, you need to check if it's currently available or not using 'see_item_stock' tool.
        * If the item is available, you can proceed with taking more details, but if it's not, send a message like "Sorry, but currently we don't have this item available", something like that.
        * To start with order creation you first need to get the name of the customer
        * After you get the name, you have to get the phone number and address of the customer.
        * Once, these three things are ready, ask for an active email.
        * After getting all the details, create an order using 'create_order' tool, to post an order into the system."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")]
    )
    order_llm= prompt | order_tool_based_model
    history= state.get("messages", [])
    current_user_message= state["incoming_message"]
    response= order_llm.invoke({"chat_history": history, "input": current_user_message})
    print("RAW LLM RESPONSE:", response, flush=True)

    # If the model suggested a tool/function call, execute the tool and use its result as the reply.
    tool_result_text = None
    try:
        # LangChain/Model responses may expose tool calls in different places depending on model wrapper.
        # Check common shapes: response.get("tool_calls"), response.get("additional_kwargs") with function_call,
        # or a top-level 'tool_call' key. Try multiple fallbacks.
        tool_calls = None
        if isinstance(response, dict):
            tool_calls = response.get("tool_calls") or response.get("tool_call")
            additional = response.get("additional_kwargs") or {}
            func_call = additional.get("function_call") if isinstance(additional, dict) else None
            if not tool_calls and func_call:
                tool_calls = [func_call]
        else:
            # response might be an object with attributes
            tool_calls = getattr(response, "tool_calls", None) or getattr(response, "tool_call", None)
            additional = getattr(response, "additional_kwargs", {})
            func_call = additional.get("function_call") if isinstance(additional, dict) else None
            if not tool_calls and func_call:
                tool_calls = [func_call]

        if tool_calls:
            # Only handle the first tool call for simplicity
            tc = tool_calls[0]
            # tc may be a dict with 'name' and 'args' JSON-string, or an object with similar attrs
            if isinstance(tc, dict):
                tool_name = tc.get("name")
                args = tc.get("args")
            else:
                tool_name = getattr(tc, "name", None)
                args = getattr(tc, "args", None)

            # args may be a JSON string or a dict
            import json
            parsed_args = {}
            if args:
                if isinstance(args, str):
                    try:
                        parsed_args = json.loads(args)
                    except Exception:
                        # fallback: try to eval (not ideal), but keep safe — treat as empty
                        parsed_args = {}
                elif isinstance(args, dict):
                    parsed_args = args

            # Map tool name to the actual callable we imported above
            tool_map = {fn.__name__: fn for fn in _order_tools}
            tool_callable = tool_map.get(tool_name)
            if tool_callable:
                # Call the tool with parsed args (assume kwargs)
                try:
                    tool_result = tool_callable(**parsed_args)
                    # If tool returns non-str, convert
                    tool_result_text = str(tool_result)
                except Exception as e:
                    tool_result_text = f"(tool execution error: {e})"
            else:
                tool_result_text = f"(no tool available named {tool_name})"
    except Exception:
        # Don't let tool-exec debug stop normal operation
        import traceback
        traceback.print_exc()

    if tool_result_text:
        answer = tool_result_text
    else:
        answer= extract_text(response)

    return {
        "messages": [AIMessage(content=answer)],
        "response_to_user": answer
    }

#-----------------Agent 3-----------------------
#-----------------Tool import ------------------
from customer_complaint import generate_ticket
rag_retreiver_tool= Rag_tool()
_complaint_tools = [generate_ticket]
ticket_llm= model.bind_tools(tools=_complaint_tools)
def customer_complaint_agent(state:BotState):
    prompt= ChatPromptTemplate.from_messages(
         [
             ("system", """You are assigned an experienced customer complaint taker, who can answer to any complaint questions regarding refund, return, or any other sort of complain.
            You can answer questions based on context provided 'rag_tool', but if the problem is beyond context level, you can proceed forward with 'generate_ticket_ tool.
            # You can take a complain by using 'generate_ticket' tool.
            * First you have to ask for user name.
            * Next, you have to get a valid active email from user.
            * After that, you have to ask for phone number from user.
            * Then you need to analyze the problem customer is facing, and write it in the format of complaint so we can later accomodate.
            After tool call is succesfull, you have to tell user you'll be accomodated within 24 hours."""),
            MessagesPlaceholder(variable_name="chat_history")
            ("human", "{input}")
         ]
    )
    ticket_agent_chain= prompt | ticket_llm
    history= state.get("messages", [])
    current_user_message= state["incoming_message"]
    response= ticket_agent_chain.invoke({"chat_history": history, "input": current_user_message})
    answer= extract_text(response)
    return {
        "messages": [AIMessage(content=answer)],
        "response_to_user": answer
    }

@app.get("/")
def health():
    return {"status": "ok"}

def routing_agent(state: BotState) -> Literal["customer_complaint_agent", "customer_query_agent", "order_management_agent"]:
    if state["classifier"] == "customer_complaint":
        return "customer_complaint_agent"
    if state["classifier"]== "customer_query":
        return "customer_query_agent"
    return "order_management_agent"

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Open once, reused across requests
_conn = sqlite3.connect("conversations.db", check_same_thread=False)
checkpointer = SqliteSaver(_conn)

def main_graph():
    graph= StateGraph(BotState)
    graph.add_node("classifier_agent", classifier_agent)
    graph.add_node("customer_query_agent", customer_query_agent)
    graph.add_node("customer_complaint_agent", customer_complaint_agent)
    graph.add_node("order_management_agent", order_management_agent)

    graph.add_edge(START, 'classifier_agent')
    graph.add_conditional_edges('classifier_agent', routing_agent)
    graph.add_edge('customer_query_agent', END)
    graph.add_edge('customer_complaint_agent', END)
    graph.add_edge('order_management_agent', END)
    return graph.compile(checkpointer=checkpointer)
    
compiled_graph= main_graph()

@wa.on_message()
def handle_text_message(client:WhatsApp, message: types.Message):
    incoming_text= extract_text_message(message)
    if not incoming_text:
        message.reply_text("Sorry, I couldn't understand that message. Could you try again?")
        return 
    wa_id = message.from_user.wa_id
    config = {"configurable": {"thread_id": str(wa_id)}}
    try:
        result= compiled_graph.invoke(
            {
                "incoming_message": incoming_text,
                "wa_id": message.from_user.wa_id,
                "messages": [HumanMessage(content=incoming_text)]
            },
            config= config,
        )
        print("DEBUG RESULT: ", result, flush=True)
        ai_reply= result.get("response_to_user") or "Sorry, I'm having trouble right now. Please try again in a moment."
    except Exception as e:
        import traceback
        traceback.print_exc()
        ai_reply = "Sorry, I'm having trouble right now. Please try again in a moment."
    message.reply_text(ai_reply)