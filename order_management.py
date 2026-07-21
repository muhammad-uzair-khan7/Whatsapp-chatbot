from langchain_core.tools import tool
import requests
import os
from dotenv import load_dotenv

load_dotenv()
POS_BASE_URL = os.getenv("POS_BASE_URL")
TIMEOUT_SECONDS = 5


def get_order_status(order_id: int) -> dict:
    """
    Fetch live order status from the POS.

    Returns a dict like:
        {"found": True, "order_id": 4821, "status": "In the Kitchen", ...}
    or on failure:
        {"found": False, "error": "Order 9999 not found"}
    """
    try:
        resp = requests.get(f"{POS_BASE_URL}/api/orders/{order_id}", timeout=TIMEOUT_SECONDS)
        if resp.status_code == 404:
            return {"found": False, "error": resp.json().get("detail", "Order not found")}
        resp.raise_for_status()
        data = resp.json()
        data["found"] = True
        return data
    except requests.exceptions.RequestException as e:
        return {"found": False, "error": f"POS system unreachable: {e}"}


def check_menu_item_availability(item_name: str) -> dict:
    """
    Check whether a MENU item (e.g. 'Cappuccino') can currently be made,
    based on whether all of its recipe ingredients are in stock.

    NOTE: this is different from raw-ingredient inventory — a customer
    orders "Cappuccino", not "coffee_beans", so we check availability at
    the menu-item level via /api/menu/availability, not /api/inventory/check.

    Returns a dict like:
        {"found": True, "item_name": "Cappuccino", "available": True, "max_makeable": 42, "shortages": []}
    or on failure:
        {"found": False, "error": "'Unicorn Latte' is not on the menu"}
    """
    try:
        resp = requests.get(
            f"{POS_BASE_URL}/api/menu/availability/{item_name}", timeout=TIMEOUT_SECONDS
        )
        if resp.status_code == 404:
            return {"found": False, "error": resp.json().get("detail", "Item not on menu")}
        resp.raise_for_status()
        data = resp.json()
        data["found"] = True
        return data
    except requests.exceptions.RequestException as e:
        return {"found": False, "error": f"POS system unreachable: {e}"}


def create_new_order(name: str, items: list[str], source: str = "WhatsApp") -> dict:
    """
    Place a new order (e.g. a customer confirms their cart in the chatbot).

    Returns a dict like:
        {"success": True, "order_id": 4824, "total_bill_pkr": 1930, ...}
    or on failure (bad item / insufficient stock):
        {"success": False, "error": "Cannot fulfill order...", "shortages": [...]}
    """
    try:
        resp = requests.post(
            f"{POS_BASE_URL}/api/orders/create",
            json={"name": name, "items": items, "source": source},
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code == 400:
            detail = resp.json().get("detail", {})
            if isinstance(detail, dict):
                return {"success": False, "error": detail.get("message"), "shortages": detail.get("shortages", [])}
            return {"success": False, "error": str(detail)}
        if resp.status_code == 422:
            return {"success": False, "error": f"Invalid order request: {resp.json().get('detail')}"}
        resp.raise_for_status()
        data = resp.json()
        data["success"] = True
        return data
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"POS system unreachable: {e}"}


@tool
def check_order_status(order_id: int):
    """Look up the live status of a customer's cafe order by order id."""
    result = get_order_status(order_id)
    if not result["found"]:
        return f"Sorry, I couldn't find that order: {result['error']}"
    return (
        f"Order #{result['order_id']} is currently '{result['status']}'. "
        f"Items: {', '.join(result['items'])}. "
        f"Total: PKR {result['total_bill_pkr']} ({result['tax_included']})."
    )


@tool
def create_order(name: str, items: list[str]):
    """Create an order for the item(s) a customer wants to purchase."""
    result = create_new_order(name, items)
    if not result["success"]:
        shortages = result.get("shortages")
        if shortages:
            return f"Sorry, there was a problem booking your order: {', '.join(shortages)}."
        return f"Sorry, there was a problem booking your order: {result.get('error', 'unknown error')}."
    return (
        f"Order created for {result['name']} with order id {result['order_id']}, "
        f"for {', '.join(result['items'])}. "
        f"Total: PKR {result['total_bill_pkr']} ({result['tax_included']})."
    )


@tool
def see_item_stock(item_name: str):
    """Check whether a menu item (e.g. 'Cappuccino', 'Avocado Toast') is currently available to order."""
    result = check_menu_item_availability(item_name)
    if not result["found"]:
        return f"Sorry, I couldn't check that item: {result['error']}"
    if not result["available"]:
        missing = ", ".join(result.get("shortages", [])) or "an ingredient"
        return f"Sorry, we currently don't have {item_name} available (out of {missing})."
    return f"Yes, {item_name} is available right now."