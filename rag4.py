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

def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run()

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

def create_vectorstore(doc_text):
    loader = TextLoader(doc_text)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="gemma:2b")
    return FAISS.from_documents(split_docs, embeddings)

def initialize_qa_chain(vectorstore):
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
        return_source_documents=True  # Optional: Return source documents for debugging
    )
    return qa_chain

def main():
    video_path = input("Human: Provide video path: ").strip()
    audio_path = "temp_audio.wav"

    try:
        print("Machine: Extracting and transcribing audio...")
        extract_audio(video_path, audio_path)
        transcript = transcribe_audio(audio_path)
        os.remove(audio_path)
        
        print(f"Machine: Transcript:\n{transcript}")
        with open("transcript.txt", "w") as file:
            file.write(transcript)

        print("Machine: Creating vector store...")
        vectorstore = create_vectorstore("transcript.txt")

        print("Machine: Setting up chatbot...")
        qa_chain = initialize_qa_chain(vectorstore)

        while True:
            query = input("Human: Ask a question (or 'exit' to quit): ").strip()
            if query.lower() == "exit":
                print("Machine: Goodbye!")
                break
            print("Machine: Searching for answer...")
            try:
                # Use invoke to handle multiple outputs
                response = qa_chain.invoke({"query": query})
                
                # Extract the main result and optionally source documents
                result = response["result"]
                print(f"Machine: {result}")
                
                # Optional: Display source documents for debugging

                source_docs = response.get("source_documents", [])
                if source_docs:
                    print("\nMachine: Sources:")
                    for doc in source_docs:
                        print(f"- {doc.metadata.get('source', 'Unknown')}: {doc.page_content[:200]}...")
            except Exception as e:
                print(f"Machine: Error - {e}")

    except Exception as e:
        print(f"Machine: An error occurred: {e}")

if __name__ == "__main__":
    main()
