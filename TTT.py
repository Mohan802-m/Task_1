# Step 1: Install necessary libraries
# pip install moviepy SpeechRecognition pydub

import moviepy.editor as mp
import speech_recognition as sr

# Step 2: Upload video file and extract audio
def extract_audio_from_video(video_path, audio_path):
    # Load the video file
    video = mp.VideoFileClip(video_path)
    # Extract audio
    audio = video.audio
    # Write audio to file
    audio.write_audiofile(audio_path)

# Step 3: Convert audio to text
def audio_to_text(audio_path):
    # Initialize recognizer
    recognizer = sr.Recognizer()
    # Load the audio file
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    # Convert audio to text
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Sorry, there was an issue with the speech recognition service."

# Example usage
video_path = "your_video.mp4"  # Path to your video file
audio_path = "extracted_audio.wav"  # Path to save the extracted audio

# Extract audio from video
extract_audio_from_video(video_path, audio_path)

# Convert audio to text
text = audio_to_text(audio_path)
print(text)
