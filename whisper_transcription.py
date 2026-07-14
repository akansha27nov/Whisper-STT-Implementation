"""
Speech-to-Text Implementation
Author: Akansha Verma
Description: use OpenAI's Whisper API to transcribe audio files
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def verify_audio_file(file_path_str):
    file_path = Path(file_path_str)

    # 1. Check if file exists
    if not file_path.exists():
        print(f"Error: The file '{file_path}' does not exist.")
        return False
    
    if not file_path.is_file():
        print(f"Error: '{file_path}' is a directory, not a file.")
        return False

    print(f"File found at: {file_path.absolute()}")

audio_path = "audio/CA138clip.mp3"
verify_audio_file(audio_path)