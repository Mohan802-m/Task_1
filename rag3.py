import ffmpeg
import speech_recognition as sr
import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Load environment variables
load_dotenv()

# Machine: Ask for video input
print("Machine: Please provide the path to your video file.")
video_path = input("Human: ").strip()
audio_path = "temp_audio.wav"

try:
    # Step 1: Extract Audio from Video using ffmpeg
    print("Machine: Extracting audio from the video...")
    ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run()

    # Step 2: Convert Audio to Text
    print("Machine: Converting audio to text...")
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    # Clean up temporary audio file
    os.remove(audio_path)

    # Output the transcript
    print("Machine: I have transcribed the audio. Here is the transcript:\n")
    print(f"Human: {text}")

    # Save the transcript to a text file
    output_file = "transcript.txt"
    with open(output_file, "w") as file:
        file.write(text)
    print("Machine: The transcript has been saved to 'transcript.txt'.")

    # Load the transcript for processing
    print("Machine: Loading the transcript for further processing...")
    loader = TextLoader('transcript.txt')
    documents = loader.load()

    # Split transcript into chunks
    print("Machine: Splitting the transcript into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    # Create embeddings using Ollama
    ollama_embeddings = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-de")  # Replace with a valid Ollama model name

    # Create a FAISS vectorstore
    print("Machine: Creating a vector store for querying...")
    vectorstore = FAISS.from_documents(split_docs, ollama_embeddings)

    # Initialize a retrieval-augmented generator (RAG)
    print("Machine: Setting up the RAG pipeline...")
    retriever = vectorstore.as_retriever()

    # Use a Hugging Face LLM as the generator
    print("Machine: Initializing the Hugging Face LLM...")
    hf_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-125M", device=0)  # Load a suitable Hugging Face model
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Combine retriever and generator
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)

    # Chat loop
    print("Machine: The chatbot is ready! You can ask questions about the video content.")
    while True:
        print("Machine: Please enter your query, or type 'exit' to end the conversation.")
        query = input("Human: ").strip()
        if query.lower() == "exit":
            print("Machine: Thank you for chatting! Goodbye!")
            break

        print("Machine: Let me find the most relevant response...")
        try:
            response = qa_chain.run(query)
            print(f"Machine: {response}")
        except Exception as e:
            print(f"Machine: Sorry, I encountered an error: {e}")

except Exception as e:
    print(f"Machine: An error occurred: {e}")
