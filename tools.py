import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_name = os.getenv("PINECONE_INDEX_NAME","appening-rag")
    print(f"Using index: {index_name}")
    vs = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    return vs.as_retriever(search_kwargs={"k":4})
