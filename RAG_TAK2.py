import ffmpeg
import speech_recognition as sr
import os

# Path to video and temporary audio file
video_path = input("\nEnter the path to your video file: ").strip()
audio_path = "temp_audio.wav"

# Step 1: Extract Audio from Video using ffmpeg
print("Extracting audio from video...")
ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run()

# Step 2: Convert Audio to Text
print("Converting audio to text...")
recognizer = sr.Recognizer()
with sr.AudioFile(audio_path) as source:
    audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data)

# Clean up temporary audio file
os.remove(audio_path)

# Output the transcript
print("\nTranscript:\n")
print(text)

# Save the transcript to a text file
output_file = "transcript.txt"
with open(output_file, "w") as file:
    file.write(text)
#---------------------------------------
#========================================

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

