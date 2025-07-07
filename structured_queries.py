import re
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from bson import ObjectId
from datetime import datetime

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

# --- MongoDB Client ---
client = MongoClient(MONGODB_URI)
db = client["pet-store-samyotech-in"]

# Map user-friendly entity names to MongoDB collection names
COLLECTION_MAP = {
    'product': 'products',
    'products': 'products',
    'customer': 'customers',
    'customers': 'customers',
    'employee': 'employees',
    'employees': 'employees',
    'order': 'orders',
    'orders': 'orders',
    'category': 'categories',
    'categories': 'categories',
    'subcategory': 'subcategories',
    'subcategories': 'subcategories',
    'purchase': 'purchases',
    'purchases': 'purchases',
    'user': 'users',
    'users': 'users',
    'company': 'companies',
    'companies': 'companies',
    'product type': 'additemmodels',
    'product types': 'additemmodels',
    'pet type': 'pettypemodels',
    'pet types': 'pettypemodels',
    'package model': 'packagemodels',
    'package models': 'packagemodels',
    'registration model': 'registrationmodels',
    'registration models': 'registrationmodels',
    'setting': 'settings',
    'settings': 'settings',
}

# Patterns for intent detection
COUNT_PATTERNS = [
    r"how many (.+?)\?",
    r"count (all )?(.+?)$",
    r"what(?:'s| is) the total number of (.+?)\?",
]
LIST_PATTERNS = [
    r"list (all )?(.+?)$",
    r"show me (all )?(.+?)$",
    r"display (all )?(.+?)$",
    r"get (all )?(.+?)$",
]

def clean_entity(entity):
    # Remove common trailing phrases
    entity = entity.lower().strip()
    for phrase in [
        "have been placed", "are there", "exist", "are registered", "are available",
        "are in the store", "in the store", "in each category", "with their categories",
        "details", "profiles", "information", "models", "types"
    ]:
        if entity.endswith(phrase):
            entity = entity[: -len(phrase)].strip()
    # Remove leading/trailing stopwords and punctuation
    entity = entity.strip(" ?.")
    # Normalize common plurals/singulars
    if entity.endswith('ies'):
        entity = entity[:-3] + 'y'  # categories -> category
    elif entity.endswith('s') and not entity.endswith('ss'):
        # Only remove 's' if not 'ss' (e.g., 'address')
        if entity not in ['orders', 'users', 'customers', 'employees', 'categories', 'subcategories', 'purchases', 'companies', 'settings', 'products']:
            entity = entity[:-1]
    # Special case: if entity is 'product', map to 'products'
    if entity == 'product':
        entity = 'products'
    if entity == 'order':
        entity = 'orders'
    if entity == 'category':
        entity = 'categories'
    if entity == 'subcategory':
        entity = 'subcategories'
    if entity == 'purchase':
        entity = 'purchases'
    if entity == 'user':
        entity = 'users'
    if entity == 'company':
        entity = 'companies'
    return entity

def detect_intent(question):
    q = question.lower().strip()
    # Relational: subcategories under a category
    m = re.search(r"subcategories.*under (?:the )?(.+?) categor", q)
    if m:
        category_name = m.group(1).strip()
        return ('list_subcategories_by_category', category_name)
    # Order by user
    m = re.search(r"orders? (?:for|of|by) user (.+)", q)
    if m:
        user_info = m.group(1).strip()
        return ('list_orders_by_user', user_info)
    # Order by order ID
    m = re.search(r"order details? (?:for|of)? ?order id ([\w-]+)", q)
    if m:
        order_id = m.group(1).strip()
        return ('order_by_id', order_id)
    # Orders by date
    m = re.search(r"orders? (?:on|for|placed on) (\d{4}-\d{2}-\d{2})", q)
    if m:
        date = m.group(1).strip()
        return ('orders_by_date', date)
    # Orders by date range
    m = re.search(r"orders? (?:between|from) (\d{4}-\d{2}-\d{2}) (?:and|to) (\d{4}-\d{2}-\d{2})", q)
    if m:
        start_date, end_date = m.group(1).strip(), m.group(2).strip()
        return ('orders_by_date_range', (start_date, end_date))
    # Count
    for pat in COUNT_PATTERNS:
        m = re.search(pat, q)
        if m:
            entity = m.group(1) if m.lastindex == 1 else m.group(m.lastindex)
            entity = clean_entity(entity)
            return ('count', entity)
    # List
    for pat in LIST_PATTERNS:
        m = re.search(pat, q)
        if m:
            entity = m.group(2) if m.lastindex >= 2 else m.group(m.lastindex)
            entity = clean_entity(entity)
            return ('list', entity)
    # Add more intent detection as needed
    return (None, None)

def format_product(doc):
    # Show only key fields for products
    return (
        f"Name: {doc.get('productName', 'N/A')}, "
        f"Price: {doc.get('price', 'N/A')}, "
        f"Original Price: {doc.get('originalPrice', 'N/A')}, "
        f"Discount: {doc.get('discount', 'N/A')}, "
        f"Quantity: {doc.get('quantity', 'N/A')}"
    )

def format_order(doc):
    return (
        f"OrderID: {doc.get('_id', 'N/A')}, "
        f"User: {doc.get('user', {}).get('name', doc.get('userId', 'N/A'))}, "
        f"Date: {doc.get('createdAt', 'N/A')}, "
        f"Total: {doc.get('amount', doc.get('total', 'N/A'))}, "
        f"Status: {doc.get('orderStatus', doc.get('status', 'N/A'))}"
    )

    

def format_generic(doc):
    # Try to show a name or main field, else show a summary
    for key in ['name', 'productName', 'title', 'username', 'email']:
        if key in doc:
            return f"{key.capitalize()}: {doc[key]}"
    # Fallback: show first 2-3 fields
    items = list(doc.items())
    summary = ', '.join(f"{k}: {v}" for k, v in items[:3])
    return summary

def handle_structured_query(intent, entity):
    entity = clean_entity(entity)
    # Relational: subcategories under a category
    if intent == 'list_subcategories_by_category':
        # Find the category
        cat = db['categories'].find_one({'name': {'$regex': f'^{entity}$', '$options': 'i'}})
        if not cat:
            return f"No category found with name '{entity}'."
        cat_id = cat['_id']
        subcats = list(db['subcategories'].find({'categoryId': cat_id}, {'_id': 0, 'name': 1}))
        if not subcats:
            return f"No subcategories found under '{entity}'."
        lines = [f"{i+1}. {sc['name']}" for i, sc in enumerate(subcats)]
        return f"Subcategories under '{entity}' category:\n" + "\n".join(lines)
    # Orders by user
    if intent == 'list_orders_by_user':
        user = db['users'].find_one({'$or': [
            {'name': {'$regex': entity, '$options': 'i'}},
            {'email': {'$regex': entity, '$options': 'i'}}
        ]})
        if not user:
            return f"No user found matching '{entity}'."
        user_id = user['_id']
        orders = list(db['orders'].find({'userId': user_id}))
        if not orders:
            return f"No orders found for user '{entity}'."
        lines = [f"{i+1}. {format_order(o)}" for i, o in enumerate(orders)]
        return f"Orders for user '{entity}':\n" + "\n".join(lines)
    # Order by order ID
    if intent == 'order_by_id':
        try:
            order = db['orders'].find_one({'_id': ObjectId(entity)})
        except Exception:
            order = db['orders'].find_one({'_id': entity})
        if not order:
            return f"No order found with ID '{entity}'."
        return "Order details:\n" + format_order(order)
    # Orders by date
    if intent == 'orders_by_date':
        try:
            date_obj = datetime.strptime(entity, '%Y-%m-%d')
            start = datetime(date_obj.year, date_obj.month, date_obj.day)
            end = datetime(date_obj.year, date_obj.month, date_obj.day, 23, 59, 59, 999000)
            orders = list(db['orders'].find({'createdAt': {'$gte': start, '$lte': end}}))
        except Exception:
            return "Invalid date format. Use YYYY-MM-DD."
        if not orders:
            return f"No orders found on {entity}."
        lines = [f"{i+1}. {format_order(o)}" for i, o in enumerate(orders)]
        return f"Orders on {entity}:\n" + "\n".join(lines)
    # Orders by date range
    if intent == 'orders_by_date_range':
        start_date, end_date = entity
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            orders = list(db['orders'].find({'createdAt': {'$gte': start, '$lte': end}}))
        except Exception:
            return "Invalid date format. Use YYYY-MM-DD."
        if not orders:
            return f"No orders found between {start_date} and {end_date}."
        lines = [f"{i+1}. {format_order(o)}" for i, o in enumerate(orders)]
        return f"Orders from {start_date} to {end_date}:\n" + "\n".join(lines)
    # Try direct match
    collection = COLLECTION_MAP.get(entity)
    # Try plural/singular fallback
    if not collection:
        if entity.endswith('s'):
            collection = COLLECTION_MAP.get(entity[:-1])
        elif not entity.endswith('s'):
            collection = COLLECTION_MAP.get(entity + 's')
    if not collection:
        return f"Sorry, I can't find information for '{entity}'."
    if intent == 'count':
        count = db[collection].count_documents({})
        return f"There are {count} {collection} in the store."
    elif intent == 'list':
        docs = list(db[collection].find({}, {'_id': 0}))
        if not docs:
            return f"No {collection} found."
        preview = docs[:10]
        lines = []
        if collection == 'products':
            for i, doc in enumerate(preview, 1):
                lines.append(f"{i}. {format_product(doc)}")
        elif collection == 'orders':
            for i, doc in enumerate(preview, 1):
                lines.append(f"{i}. {format_order(doc)}")
        else:
            for i, doc in enumerate(preview, 1):
                lines.append(f"{i}. {format_generic(doc)}")
        more = f"\n...and {len(docs)-10} more." if len(docs) > 10 else ""
        return f"Here are the {collection} in the store:\n" + "\n".join(lines) + more
    return "Sorry, I can't handle that type of query yet." 