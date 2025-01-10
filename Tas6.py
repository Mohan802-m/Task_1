from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example to load and embed documents
loader = TextLoader('transcript.txt')
documents = loader.load()

# Split documents and embed
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# Create a FAISS vector store
faiss = FAISS.from_documents(split_docs, embeddings)

# Query the vector store
query = "What is the meaning of life?"
results = faiss.similarity_search(query)

for result in results:
    print(result)
