import os
from dotenv import load_dotenv
load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load documents
loader = TextLoader('transcript.txt')
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# Use an available Ollama embedding model
ollama_embeddings = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-de")  # Replace with a valid Ollama model name

# Create a FAISS vectorstore
vectorstoredb = FAISS.from_documents(split_docs, ollama_embeddings)

# Query from the vectorstore
# Ask the user to input a query
query = input("Please enter your query: ")
print("Result--------")

# Perform the similarity search
result = vectorstoredb.similarity_search(query)
# Check if results are found
if result:
    print("Response:", result[0].page_content)
else:
    print("Response: No records found.")

