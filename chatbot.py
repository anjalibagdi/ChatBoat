from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from dotenv import load_dotenv
from langdetect import detect
import os
from structured_queries import detect_intent, handle_structured_query
from pymongo import MongoClient

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- MongoDB Client ---
client = MongoClient(MONGODB_URI)
db = client["pet-store-samyotech-in"]


# --- RAG Functions (as before) ---
def load_vector_stores_and_retrievers():
    client = MongoClient(MONGODB_URI)
    db = client["pet-store-samyotech-in"]

    # Get all valid collections to query
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
        name: vs.as_retriever(search_kwargs={"k": 3})
        for name, vs in vector_stores.items()
    }

    print(f"🎯 Total retrievers loaded: {len(retrievers)}")

    return retrievers

def aggregate_context(question, retrievers):
    all_docs = []
    for collection_name, retriever in retrievers.items():
        try:
            docs = retriever.get_relevant_documents(question)
            for doc in docs:
                # Optionally add metadata so we know where it came from
                doc.metadata["collection"] = collection_name
                all_docs.append(doc)
        except Exception as e:
            print(f"❌ Error retrieving from {collection_name}: {e}")
    return all_docs

def run_rag_chain(question, docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not docs:
        return "No relevant documents found."

    # Build temporary retriever from the relevant docs
    temp_vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = temp_vectorstore.as_retriever()

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    result = rag_chain({"query": question})
    answer = result["result"].replace('*', '').replace('**', '').replace('\n\n', '\n')
    print("answer", answer)
    return answer
# --- Main Chatbot Entry ---
def get_response(question, session_id):
    try:
        retrievers = load_vector_stores_and_retrievers()
        print("🔍 Retriever Dictionary Keys & Types:")
        if not retrievers:
            print("⚠️ No retrievers loaded!")

        docs = aggregate_context(question, retrievers)
        answer = run_rag_chain(question, docs)
        print("answer========>>", answer)
        return answer
    except Exception as e:
        print(f"Error in get_response: {e}")
        error_msg = "Mujhe yeh information nahi hai." if 'lang' in locals() and lang == 'hi' else "I don't have this information."
        return error_msg, str(e)