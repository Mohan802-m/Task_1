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
