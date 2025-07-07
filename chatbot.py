from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from dotenv import load_dotenv
from langdetect import detect
import os
# Import structured query functions
from structured_queries import detect_intent, handle_structured_query
from pymongo import MongoClient

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Prompt templates
hindi_prompt_template = """
You are a helpful chatbot for a pet store. Answer in Hindi, be polite, and always include product name, price, and discount if available. If you don't know the answer, say "Mujhe yeh information nahi hai, kya main aapki aur kisi tarah se madad kar sakta hoon?"
Context: {context}
Question: {question}
Answer:
"""

english_prompt_template = """
You are a helpful chatbot for a pet store. Answer in English, be polite, and always include product name, price, and discount if available. If you don't know the answer, say "I don't have this information, can I assist you with something else?"
Context: {context}
Question: {question}
Answer:
"""

# --- Structured Query Support ---

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

# Add more patterns as needed for details, analytics, etc.

# --- MongoDB Client ---
client = MongoClient(MONGODB_URI)
db = client["pet-store-samyotech-in"]


# --- RAG Functions (as before) ---
def load_vector_stores_and_retrievers():
    client = MongoClient(MONGODB_URI)
    db = client["pet-store-samyotech-in"]
    collections_to_query = [col for col in db.list_collection_names() if col != "chat_history" and not col.startswith("system.") and not col.endswith("_embeddings")]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_stores = {}
    for collection in collections_to_query:
        path = f"vector_store/{collection}_faiss_index/index.faiss"
        if not os.path.exists(path):
            continue
        try:
            vector_stores[collection] = FAISS.load_local(
                f"vector_store/{collection}_faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"⚠️ Could not load FAISS index for {collection}: {e}")
    retrievers = {name: vs.as_retriever(search_kwargs={"k": 3}) for name, vs in vector_stores.items()}
    return retrievers

def select_prompt_template(question):
    lang = detect(question)
    is_hindi = lang == 'hi'
    prompt_template = hindi_prompt_template if is_hindi else english_prompt_template
    return prompt_template, lang

def aggregate_context(question, retrievers):
    all_context = []
    for collection_name, retriever in retrievers.items():
        try:
            docs = retriever.get_relevant_documents(question)
            for doc in docs:
                context_str = f"Collection: {collection_name}, Text: {doc.page_content}"
                fields = doc.metadata.get("original_fields", {})
                context_str += f", Details: {fields}"
                all_context.append(context_str)
        except Exception as e:
            print(f"Error retrieving from {collection_name}: {e}")
    context = "\n".join(all_context) if all_context else "No relevant context found."
    return context

def run_rag_chain(question, context, prompt_template, retriever):
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    result = rag_chain({"query": question, "context": context})
    answer = result["result"].replace('*', '').replace('**', '').replace('\n\n', '\n')
    return answer

# --- Main Chatbot Entry ---
def get_response(question, session_id):
    try:
        # 1. Try structured query first
        intent, entity = detect_intent(question)
        if intent and entity:
            answer = handle_structured_query(intent, entity)
            # Store conversation in history
            chat_history = MongoDBChatMessageHistory(
                session_id=session_id,
                connection_string=MONGODB_URI,
                database_name="pet-store-samyotech-in",
                collection_name="chat_history"
            )
            chat_history.add_user_message(question)
            chat_history.add_ai_message(answer)
            return answer, f"Structured query: {intent} {entity}"
        # 2. Fallback to RAG
        retrievers = load_vector_stores_and_retrievers()
        if not retrievers:
            return "No knowledge base available.", "No retrievers loaded."
        prompt_template, lang = select_prompt_template(question)
        context = aggregate_context(question, retrievers)
        first_retriever = list(retrievers.values())[0]
        answer = run_rag_chain(question, context, prompt_template, first_retriever)
        # Store conversation in history
        chat_history = MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=MONGODB_URI,
            database_name="pet-store-samyotech-in",
            collection_name="chat_history"
        )
        chat_history.add_user_message(question)
        chat_history.add_ai_message(answer)
        return answer, context
    except Exception as e:
        print(f"Error in get_response: {e}")
        error_msg = "Mujhe yeh information nahi hai." if 'lang' in locals() and lang == 'hi' else "I don't have this information."
        return error_msg, str(e)