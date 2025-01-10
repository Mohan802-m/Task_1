import streamlit as st
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

## Langsmith Tracking

# Load environment variables
load_dotenv()

# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_PROJECT_Ollama_QA"]=os.getenv("LANGCHAIN_PROJECT_Ollama_QA")


# Streamlit UI
st.title("Video-Based Chatbot with LangChain")
st.write("Upload a video file, and I'll help answer your questions based on its content!")

# Upload video
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
if uploaded_video:
    # Save video to a temporary file
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Extract audio
   # st.write("Extracting audio from the video...")
    audio_path = "temp_audio.wav"
    ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run()


    # Transcribe audio
    #st.write("Transcribing audio...")
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data)

    # Save the transcript
    os.remove(audio_path)
    with open("transcript.txt", "w") as file:
        file.write(transcript)
    #st.write("**Transcript:**")
    #st.text(transcript)

    # Create vector store
    #st.write("Creating vector store...")
    loader = TextLoader("transcript.txt")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="gemma:2b")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Set up chatbot
    #st.write("Setting up chatbot...")
    retriever = vectorstore.as_retriever()

    prompt_template = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = OllamaLLM(model="gemma:2b")

    qa_chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        chain_type="stuff",
        llm=llm,
    )

    # Chatbot interaction
    st.write("### Chat with the Bot")
    query = st.text_input("Ask a question:")
    if query:
        st.write("Searching for the answer...")
        try:
            response = qa_chain.invoke({"query": query})
            result = response["result"]
            st.write(f"**Answer:** {result}")
        except Exception as e:
            st.error(f"Error: {e}")

    # Cleanup temporary files
    os.remove(video_path)
    os.remove("transcript.txt")
