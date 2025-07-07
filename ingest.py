from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize MongoDB client
client = MongoClient(MONGODB_URI)
db = client["pet-store-samyotech-in"]

# Fields to exclude from embedding
EXCLUDED_FIELDS = {"_id", "createdAt", "updatedAt", "__v", "isDelete", "image"}

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Get all collections dynamically (exclude system collections, chat_history, and _embeddings collections)
collections = [col for col in db.list_collection_names() if col != "chat_history" and not col.startswith("system.") and not col.endswith("_embeddings")]

# Loop through each collection
for collection_name in collections:
    print(f"\n🔄 Processing collection: {collection_name}")
    source_collection = db[collection_name]
    target_collection = db[f"{collection_name}_embeddings"]

    # Get fields from a sample document
    sample_doc = source_collection.find_one()
    if not sample_doc:
        print(f"⚠️ No documents found in {collection_name}")
        continue
    field_names = [key for key in sample_doc.keys() if key not in EXCLUDED_FIELDS]
    if not field_names:
        print(f"⚠️ No valid fields found in {collection_name}")
        continue
    print(f"🔍 Embedding fields: {', '.join(field_names)}")

    # Load documents
    documents = []
    cursor = source_collection.find({field_name: {"$exists": True} for field_name in field_names})
    for doc in cursor:
        # Concatenate all field values into a single string
        field_values = []
        for field_name in field_names:
            value = doc.get(field_name)
            if value is not None:
                field_values.append(str(value))  # Convert to string
            else:
                field_values.append("")
        text = " ".join(field_values)
        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source_id": str(doc["_id"]),
                    "source_collection": collection_name,
                    "original_fields": {field_name: doc.get(field_name) for field_name in field_names}
                }
            )
        )

    if not documents:
        print(f"⚠️ No valid data found in {collection_name}")
        continue

    try:
        # Generate embeddings and update/insert in MongoDB
        embedding_vectors = embeddings.embed_documents([doc.page_content for doc in documents])
        for doc, embedding in zip(documents, embedding_vectors):
            # Check if embedding exists for this source_id
            existing_doc = target_collection.find_one({"metadata.source_id": doc.metadata["source_id"]})
            if existing_doc:
                # Update existing embedding
                target_collection.update_one(
                    {"metadata.source_id": doc.metadata["source_id"]},
                    {
                        "$set": {
                            "text": doc.page_content,
                            "metadata": doc.metadata,
                            "embedding": embedding
                        }
                    }
                )
                print(f"🔄 Updated embedding for document {doc.metadata['source_id']} in '{collection_name}_embeddings'")
            else:
                # Insert new embedding
                target_collection.insert_one({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": embedding
                })
                print(f" Inserted new embedding for document {doc.metadata['source_id']} in '{collection_name}_embeddings'")

        # Create and save FAISS index (overwrite existing)
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(f"vector_store/{collection_name}_faiss_index")
        print(f"✅ Stored/Updated {len(documents)} embeddings in MongoDB '{collection_name}_embeddings' and FAISS 'vector_store/{collection_name}_faiss_index'")
    except Exception as e:
        print(f"❌ Error while embedding '{collection_name}': {e}")