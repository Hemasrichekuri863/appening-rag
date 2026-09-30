import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "appening-rag")
PDF_PATH = "data/Ebook-Agentic-AI.pdf"

def main():
    print(f"Loading PDF from {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Delete old index if exists to fix dimension
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"Deleting old index {INDEX_NAME}...")
        pc.delete_index(INDEX_NAME)
        import time
        time.sleep(10)

    print(f"Creating index {INDEX_NAME} with dimension 384...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    # Wait for index ready
    import time
    print("Waiting for index to be ready...")
    time.sleep(20)

    print("Uploading vectors...")
    from langchain_pinecone import PineconeVectorStore
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print(f"Done! Ingested {len(chunks)} chunks!")

if __name__ == "__main__":
    main()