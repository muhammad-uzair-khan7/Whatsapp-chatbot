# WhatsApp Chatbot

This project demonstrates a full end-to-end WhatsApp AI chatbot for cafes, restaurants, hotels, salons, or any small-to-medium business that handles a high volume of repetitive customer messages. It's an **Agentic AI** based project, built around three specialized agents that collaborate to handle order management, customer queries, and customer complaints — all through a business's existing WhatsApp number.

## Tech stack

* **Meta WhatsApp Cloud API** — messaging channel
* **PyWa** — Python wrapper for the WhatsApp Cloud API, handles webhook registration and message send/receive
* **LangChain** — LLM orchestration, prompt templates, tool binding, RAG
* **LangGraph** — agent graph orchestration, conversation state, routing between agents, persistent memory (checkpointing)
* **Pydantic** — structured output validation (classification, request/response schemas)
* **FastAPI** — web server for both the WhatsApp webhook and the POS API
* **Custom POS System** — a mock/real point-of-sale backend exposing orders, inventory, and complaint-ticket endpoints
* **Google Gemini** (`gemini-2.5-flash`) — primary LLM powering classification and all three agents
* **OpenRouter (OpenAI-compatible API)** — speech-to-text and image-to-text for voice notes and images sent by customers
* **SQLite** — persistent conversation memory via LangGraph's checkpointer

## What problem does this solve

Are you a restaurant or café owner who gets flooded with messages like *"What's the price of the Spanish Latte?"*, *"How long does delivery take?"*, or has to personally handle every customer complaint? This bot automates exactly that kind of repetitive, low-complexity conversation — so your time goes toward the parts of the business that actually need a human, instead of answering the same handful of questions over and over.

Concretely, the bot can:
- Answer menu, pricing, and general café-info questions using a retrieval-augmented (RAG) knowledge base built from your own café details document.
- Take a customer through the full order flow — item selection, availability check, customer details, and order creation — directly through chat, and confirm the order against a live POS system.
- Check the live status of an existing order by order ID.
- Log a customer complaint end-to-end (name, email, phone, issue description) into both the POS system and a Google Sheet, so nothing falls through the cracks.
- Understand voice notes and images sent by customers, not just text.
- Reply in Roman English (i.e. Urdu written in Latin script), since that's how many customers actually type.

## How this project was built

We mapped the project and broke it into distinct modules before writing any orchestration code. We deliberately chose an **Agentic AI framework** over a single monolithic AI agent — a customer complaint has very different requirements (structured data collection, ticket logging) than a menu question (retrieval + short factual answers) or an order (multi-turn state, tool calls against a live inventory system). Trying to handle all three with one prompt and one tool set gets messy fast; separating them into dedicated agents with their own prompts, tools, and completion criteria kept each one focused and easier to reason about.

To achieve this, we used `LangChain` for the individual LLM chains, prompt templates, and tool binding, and `LangGraph` to wire the agents together into a single stateful graph — including routing between agents, looping back when a conversation goes off-topic, and persisting conversation history across messages and server restarts.

## Architecture

### The three agents

1. **`classifier_agent`** — the entry point for any new or ambiguous conversation. Uses Gemini with structured output (a Pydantic schema) to classify the incoming message into one of: `order_management`, `customer_query`, or `customer_complaint`.

2. **`customer_query_agent`** — answers general questions using a RAG pipeline. Café details are chunked, embedded (`sentence-transformers/all-MiniLM-L6-v2`), and stored in a local FAISS vector index. Every incoming query retrieves the most relevant chunks and answers grounded in that context, rather than letting the model improvise.

3. **`order_management_agent`** — a tool-using agent bound to three tools:
   - `see_item_stock` — checks whether a menu item can currently be made, based on live ingredient inventory
   - `check_order_status` — looks up an existing order by ID
   - `create_order` — places a new order once all required customer details are collected

   This agent runs a full tool-execution loop: it can call a tool, receive the result, and use that result to decide its next message or next tool call, rather than being limited to one tool call per turn.

4. **`customer_complaint_agent`** — walks the customer through providing their name, email, phone number, and complaint description, then calls `generate_ticket` to log the complaint into the POS system and a Google Sheet.

### Sticky routing

Once a conversation enters `order_management_agent` or `customer_complaint_agent`, the graph stays "stuck" on that agent for subsequent messages — it doesn't re-run the classifier on every turn, which would waste an LLM call and risk misclassifying a mid-flow message (e.g. a bare order ID like `"4823"` being misread as something else). Each agent tracks its own completion state and releases control back to the classifier once its task is done (an order is created, or a complaint is logged).

If a customer goes off-topic mid-flow (e.g. asks about opening hours in the middle of placing an order), the active agent recognizes this and signals a re-route, sending the conversation back through the classifier so it lands on the right agent — without losing the conversation's context.

### Conversation memory

Conversation history is persisted per WhatsApp user via LangGraph's `SqliteSaver` checkpointer, keyed by the user's WhatsApp ID as the `thread_id`. This means conversations survive server restarts, and each customer's chat history is kept separate and private to their own thread.

### Media handling

Voice notes and images sent by customers are converted to text before being handed to the graph:
- **Voice notes** are downloaded and transcribed via an OpenRouter-hosted speech-to-text model.
- **Images** are downloaded and either OCR'd or described via an OpenRouter-hosted vision model, combined with any caption text the customer included.

## Project structure

```
whatsapp-agent/
├── whatsapp_api.py        # Main bot: FastAPI app, LangGraph graph, agent definitions, webhook handler
├── order_management.py    # Order-related tools (create_order, check_order_status, see_item_stock)
├── customer_complaint.py  # Complaint tool (generate_ticket), Google Sheets logging
├── cafe_details.txt       # Source document for the RAG knowledge base
├── credentials.json       # Google service account credentials (for Sheets access)
├── .env                   # Environment variables (see below)
└── pos_server/
    └── main.py             # Separate FastAPI app simulating/serving the café's POS system
```

## Setup

### 1. Environment

```bash
uv venv
uv sync
```

(Or, without `uv`: create a standard virtual environment and `pip install` the packages listed under Tech Stack, plus `fastapi`, `uvicorn`, `pywa`, `python-dotenv`, `langgraph-checkpoint-sqlite`, `faiss-cpu`, and `gspread`.)

### 2. Environment variables (`.env`)

```
PHONENUMBER_ID=your_whatsapp_phone_id
ACCESS_TOKEN=your_whatsapp_access_token
APP_ID=your_meta_app_id
APP_SECRET=your_meta_app_secret
VERIFY_TOKEN=your_webhook_verify_token
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openrouter_api_key
POS_BASE_URL=http://127.0.0.1:8001
```

### 3. Google Sheets access

Complaint tickets are logged to a Google Sheet via `gspread`. Share the target sheet with the `client_email` found inside your `credentials.json` service account file, giving it Editor access, or the complaint agent will fail to log tickets.

## Running the project

This project runs as **two separate servers**, which need to be running at the same time.

**Start the POS server first**, from its own folder:
```bash
cd pos_server
uvicorn main:app --port 8001
```

**Then start the bot**, from the project root:
```bash
uvicorn whatsapp_api:app --port 8000
```

**Expose the bot to the internet** so Meta can reach your webhook (for local development):
```bash
ngrok http 8000
```
Make sure the `callback_url` passed into the `WhatsApp(...)` client in `whatsapp_api.py`, the URL shown by ngrok, and the webhook URL registered in the Meta App Dashboard all match exactly — ngrok's free tier issues a new URL on every restart, so this needs re-checking each time you restart the tunnel.

## Known limitations / roadmap

- **Task queue**: message processing currently runs on a background thread pool within the same process. For production traffic, this should move to a proper task queue (e.g. Redis + RQ/Celery) so work survives process restarts and can scale across multiple machines.
- **Async I/O**: HTTP calls to the POS system and Google Sheets are currently synchronous. An async rewrite (`httpx` instead of `requests`, async Sheets client) would improve concurrency under real multi-user load.
- **Model latency**: response times depend heavily on the chosen Gemini model and the number of sequential LLM calls per turn (classification, agent response, and any follow-up call after a tool result). A stable, non-preview flash-tier model is recommended for production use over preview models, which can carry significantly higher latency.
- **Rate limits**: Gemini's free tier is capped at a low daily request count, which is easy to exhaust during active testing since a single customer message can trigger multiple model calls. A paid tier is recommended before real deployment.