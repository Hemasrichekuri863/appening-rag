import os
os.makedirs("src", exist_ok=True)

# 1. ingestion.py
with open("src/ingestion.py","w", encoding="utf-8") as f:
    f.write('''import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating index {INDEX_NAME}...")
        pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    print("Ingesting to Pinecone...")
    PineconeVectorStore.from_documents(documents=chunks, embedding=embeddings, index_name=INDEX_NAME)
    print(f"Done! Ingested {len(chunks)} chunks to {INDEX_NAME}")
if __name__ == "__main__":
    main()
''')

# 2. config.py
with open("src/config.py","w", encoding="utf-8") as f:
    f.write('''import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "appening-rag")
''')

# 3. tools.py
with open("src/tools.py","w", encoding="utf-8") as f:
    f.write('''import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools import tool
load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX", "appening-rag")
@tool
def retrieve_context(query: str) -> str:
    """Retrieve relevant context from Agentic AI ebook to answer user question."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    docs = vectorstore.similarity_search(query, k=4)
    return "\\n\\n".join([d.page_content for d in docs])
''')

# 4. agent.py
with open("src/agent.py","w", encoding="utf-8") as f:
    f.write('''from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.tools import retrieve_context
def get_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [retrieve_context]
    agent = create_react_agent(llm, tools)
    return agent
''')

# 5. app.py
with open("app.py","w", encoding="utf-8") as f:
    f.write('''from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from src.agent import get_agent
app = FastAPI(title="Appening AI RAG")
agent = get_agent()
class Query(BaseModel):
    question: str
@app.get("/")
def home():
    return {"message": "Appening AI RAG is running!"}
@app.post("/chat")
def chat(q: Query):
    result = agent.invoke({"messages": [("user", q.question)]})
    last_msg = result["messages"][-1].content
    return {"answer": last_msg}
''')

# 6. __init__.py
with open("src/__init__.py","w") as f:
    f.write("")

print("ALL FILES FIXED SUCCESSFULLY!")