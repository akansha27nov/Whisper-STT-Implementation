"""
Speech-to-Text Implementation
Author: Akansha Verma
Description: use OpenAI's Whisper API to transcribe audio files
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

# ======================================================
# Step 1: Setting Up and Loading api key
# ======================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ======================================================
# Step 2: Verify downloaded file is accessible
# ======================================================
AUDIO_DATA_DIR = Path("audio")
def verify_audio_file(file_path_str):
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"Error: The file '{file_path}' does not exist.")
        return False
    
    if not file_path.is_file():
        print(f"Error: '{file_path}' is a directory, not a file.")
        return False

    print(f"File found at: {file_path.absolute()}")

audio_path = AUDIO_DATA_DIR / "CA138clip.mp3"
verify_audio_file(audio_path)

# ======================================================
# Step 3: Basic Transcription (Without Chunking)
# ======================================================
def transcribe_audio_file_as_is():
    
    with open(audio_path, "rb") as audio_file:
        print("🤖 Transcribing with Whisper...")
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript

transcription_text = transcribe_audio_file_as_is()
print("\n📝 Transcription:")
print("-" * 40)
print(transcription_text.text)
print("-" * 40)

# ======================================================
# Step 4: Transcription with Prompts (Guided Approach)
# ======================================================
