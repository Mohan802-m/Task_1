import ffmpeg
import speech_recognition as sr
import os

def video_to_text(video_path):
    """Convert video to text by extracting and processing audio."""
    audio_path = "temp_audio.wav"
    try:
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
        return text
    except sr.UnknownValueError:
        return "Could not understand the speech in the video."
    except sr.RequestError as e:
        return f"Speech recognition service error: {e}"
    except Exception as e:
        return f"Error processing the video: {e}"

def chatbot():
    """Chatbot loop to interact with the user."""
    print("Welcome to the Video-to-Text Chatbot!")
    print("Enter 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter the path to your video file: ").strip()
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        if not os.path.exists(user_input):
            print("File not found. Please try again.")
            continue
        
        print("Processing your video. Please wait...")
        transcript = video_to_text(user_input)
        print("\nTranscript:\n")
        print(transcript)

# Run the chatbot
if __name__ == "__main__":
    try:
        chatbot()
    except Exception as e:
        print(f"An error occurred: {e}")
