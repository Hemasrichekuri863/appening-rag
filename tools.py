import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools import tool

load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "appening-rag")

@tool
def retrieve_context(query: str) -> str:
    """Retrieve relevant context from Agentic AI ebook to answer user question."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    docs = vectorstore.similarity_search(query, k=4)
    return "\n\n".join([d.page_content for d in docs])
