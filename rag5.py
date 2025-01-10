import ffmpeg
import speech_recognition as sr
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT_Ollama"]=os.getenv("LANGCHAIN_PROJECT_Ollama")


# Step 1: Get the video path
video_path = input("Human: Provide video path: ").strip()

# Step 2: Extract audio from the video
print("Machine: Extracting audio from the video...")
audio_path = "temp_audio.wav"
ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run()

# Step 3: Transcribe the audio
print("Machine: Transcribing audio...")
recognizer = sr.Recognizer()
with sr.AudioFile(audio_path) as source:
    audio_data = recognizer.record(source)
    transcript = recognizer.recognize_google(audio_data)

# Save the transcript to a file and clean up temporary audio
os.remove(audio_path)
print(f"Machine: Transcript:\n{transcript}")
with open("transcript.txt", "w") as file:
    file.write(transcript)

# Step 4: Create a vector store
print("Machine: Creating vector store...")
loader = TextLoader("transcript.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="gemma:2b")
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Step 5: Set up the chatbot
print("Machine: Setting up chatbot...")
retriever = vectorstore.as_retriever()

# Define a template for responses
prompt_template = """
Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Use OllamaLLM for the LLM
llm = OllamaLLM(model="gemma:2b")

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    chain_type="stuff",
    llm=llm,
    #return_source_documents=True  # Optional: Return source documents for debugging
)

# Step 6: Start chatbot interaction
print("Machine: Chatbot is ready. You can ask questions now.")
while True:
    query = input("Human: Ask a question (or 'exit' to quit): ").strip()
    if query.lower() == "exit":
        print("Machine: Goodbye!")
        break

    print("Machine: Searching for answer...")
    try:
        response = qa_chain.invoke({"query": query})
        result = response["result"]
        print(f"Machine: {result}")
    except Exception as e:
        print(f"Machine: Error - {e}")


