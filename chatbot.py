from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from dotenv import load_dotenv
from pymongo import MongoClient
import os
import datetime
from datetime import date
from dateutil import parser
import re

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# MongoDB Client
client = MongoClient(MONGODB_URI)
db = client["pet-store-samyotech-in"]

# -------------------------------
# ✅ Intent Detection Function
# -------------------------------
def detect_collection_name(question):
    question = question.lower()
    keywords_to_collections = {
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
        'package model': 'packagemodels',
        'package models': 'packagemodels',
        'bookings': 'registrationmodels',
        'setting': 'settings',
        'settings': 'settings',
        'pet type': 'pettypemodels',
        'packages' : 'packagemodels',
    }

    for keyword, collection in keywords_to_collections.items():
        if keyword in question:
            return collection

    return None

# -------------------------------
# ✅ Updated Time Filter Function
# -------------------------------
def detect_time_filter(question):
    print("detect_time_filter:", question)
    today = date.today()
    print("today:", today)

    # Helper function to parse date safely
    def parse_date(date_value):
        if not date_value:
            print(f"Warning: createdAt is missing or None")
            return None
        try:
            if isinstance(date_value, datetime.datetime):
                return date_value.date()
            if isinstance(date_value, str):
                parsed = parser.parse(date_value, fuzzy=True)
                return parsed.date()
            print(f"Warning: createdAt is not a string or datetime: {type(date_value)}")
            return None
        except (ValueError, TypeError) as e:
            print(f"Failed to parse date: {date_value}, Error: {e}")
            return None

    question_lower = question.lower()

    if "today" in question_lower:
        return lambda d: parse_date(d.metadata.get("original_fields", {}).get("createdAt")) == today
    elif "week" in question_lower:
        start_of_week = today - datetime.timedelta(days=today.weekday())
        return lambda d: start_of_week <= parse_date(d.metadata.get("original_fields", {}).get("createdAt")) <= today
    elif "month" in question_lower:
        return lambda d: parse_date(d.metadata.get("original_fields", {}).get("createdAt")).month == today.month if parse_date(d.metadata.get("original_fields", {}).get("createdAt")) else False
    else:
        # Regular expression to match common date formats (e.g., YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY)
        date_pattern = r'\b(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4})\b'
        date_match = re.search(date_pattern, question)
        if date_match:
            matched_date = date_match.group(0)
            target_date = parse_date(matched_date)
            if target_date:
                return lambda d: parse_date(d.metadata.get("original_fields", {}).get("createdAt")) == target_date
        return None

# -------------------------------
# ✅ Load Vector Stores & Retrievers
# -------------------------------
def load_vector_stores_and_retrievers():
    client = MongoClient(MONGODB_URI)
    db = client["pet-store-samyotech-in"]

    collections_to_query = [
        col for col in db.list_collection_names()
        if col != "chat_history"
        and not col.startswith("system.")
        and not col.endswith("_embeddings")
    ]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_stores = {}

    for collection in collections_to_query:
        dir_path = f"vector_store/{collection}_faiss_index"
        index_file = os.path.join(dir_path, "index.faiss")

        if not os.path.exists(index_file):
            print(f"⚠️ FAISS index not found for collection: {collection}")
            continue

        try:
            vector_stores[collection] = FAISS.load_local(
                dir_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Loaded FAISS index for collection: {collection}")
        except Exception as e:
            print(f"❌ Could not load FAISS index for '{collection}': {e}")

    retrievers = {
        name: vs.as_retriever(search_kwargs={"k": 20})
        for name, vs in vector_stores.items()
    }

    print(f"🎯 Total retrievers loaded: {len(retrievers)}")
    return retrievers

# -------------------------------
# ✅ Aggregate Context
# -------------------------------
def aggregate_context(question, retrievers, target_collections=None):
    all_docs = []
    collections_to_use = target_collections if target_collections else retrievers.keys()
    time_filter = detect_time_filter(question)

    for collection_name in collections_to_use:
        retriever = retrievers.get(collection_name)
        if not retriever:
            continue
        try:
            docs = retriever.invoke(question)
            if docs:
                print(f"📂 Data retrieved from collection: {collection_name}")
            for doc in docs:
                doc.metadata["collection"] = collection_name
                if time_filter and not time_filter(doc):
                    print(f"Filtered out document with createdAt: {doc.metadata.get("original_fields", {}).get("createdAt")}")  # Updated debug
                    continue
                all_docs.append(doc)
        except Exception as e:
            print(f"❌ Error retrieving from {collection_name}: {e}")
    return all_docs

# -------------------------------
# ✅ Run RAG Chain
# -------------------------------
def run_rag_chain(question, docs):

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not docs:
        return "No relevant documents found."

    temp_vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = temp_vectorstore.as_retriever()

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    result = rag_chain.invoke({"query": question})
    answer = result["result"].replace('*', '').replace('**', '').replace('\n\n', '\n')
    print("answer", answer)
    return answer

# -------------------------------
# ✅ Detect Output Format
# -------------------------------
def detect_output_format(question):
    question = question.lower()
    if "table" in question:
        return "table"
    elif "list" in question:
        return "list"
    elif "json" in question:
        return "json"
    elif "text" in question or "paragraph" in question:
        return "text"
    else:
        return "default"

# -------------------------------
# ✅ Detect Required Fields
# -------------------------------
def detect_required_fields(question):
    possible_fields = ["name", "email", "phone", "mobile", "price", "category", "subcategory", "date", "status", "type", "address", "id"]

    question = question.lower()
    detected_fields = []

    for field in possible_fields:
        if field in question:
            detected_fields.append(field)

    return detected_fields if detected_fields else ["*"]  # "*" means all fields

# -------------------------------
# ✅ Main Response Function
# -------------------------------
def get_response(question, session_id):
    try:
        retrievers = load_vector_stores_and_retrievers()

        # Step 1: Detect collection
        collection = detect_collection_name(question)
        print("🧠 Collection:", collection)

        # Step 2: Detect output format
        output_format = detect_output_format(question)
        print("📋 Output Format:", output_format)

        # Step 3: Detect required fields
        fields = detect_required_fields(question)
        print("🔑 Fields:", fields)

        # Step 4: Aggregate & Generate Response
        target_collections = [collection] if collection and collection in retrievers else None
        docs = aggregate_context(question, retrievers, target_collections)
        answer = run_rag_chain(question, docs)

        return {
            "answer": answer,
        }

    except Exception as e:
        print(f"❌ Error in get_response: {e}")
        return {
            "answer": "I don't have this information.",
            "error": str(e)
        }