"""
Speech-to-Text Implementation
Author: Akansha Verma
Description: use OpenAI's Whisper API to transcribe audio files
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment

# ======================================================
# Step 1: Setting Up and Loading api key
# ======================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TRANSCRIPTION_DATA_DIR = Path("transcriptions")
AUDIO_DATA_DIR = Path("audio")
audio_path = AUDIO_DATA_DIR / "CA138clip.mp3"
prompt_for_transcription = "This is an interview about the moon landing, Maine, soil sample, kangaroo walk."

# Going forward, this filw will be divided into 2 parts:
# first is function definitions only
# second is testing
# ======================================================
# Step 2: Verify downloaded file is accessible
# ======================================================
def verify_audio_file(file_path_str):
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"Error: The file '{file_path}' does not exist.")
        return False
    
    if not file_path.is_file():
        print(f"Error: '{file_path}' is a directory, not a file.")
        return False

    print(f"File found at: {file_path.absolute()}")

# ======================================================
# Step 3: Basic Transcription (Without Chunking)
# ======================================================
def transcribe_audio_file_as_is(audio):
    
    with open(audio, "rb") as audio_file:
        print("🤖 Transcribing with Whisper...")
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    # save the result to a text file
    output_path = Path(TRANSCRIPTION_DATA_DIR/"transcript_basic.txt")
    output_path.write_text(transcript.text, encoding="utf-8")
    print(f"Unguided transcription saved to {output_path}")
    
    return transcript

# ======================================================
# Step 4: Transcription with Prompts (Guided Approach)
# ======================================================
def transcribe_audio_file_with_prompt(audio, prompt_txt):
    with open(audio, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            prompt=prompt_txt
        )
    # save the result to a text file
    output_path = Path(TRANSCRIPTION_DATA_DIR/"transcript_prompted.txt")
    output_path.write_text(transcript.text, encoding="utf-8")
    
    return transcript

# ====================================================================
# Step 5: Transcription Without Prompts (Unguided Approach) & compare
# ====================================================================

# this is same as step 3, transcription without prompt and 
# result is saved in a file.

basic_transcripted_file = TRANSCRIPTION_DATA_DIR/"transcript_basic.txt"
prompted_file = TRANSCRIPTION_DATA_DIR/"transcript_prompted.txt"

def analyse_transcripts():
    # load files
    with open(prompted_file, 'r') as f: prompted_text = f.read()
    with open(basic_transcripted_file, 'r') as f: unguided_text = f.read()
    
    # split into sentences to compare
    unguided_sentences = [s.strip() for s in unguided_text.split('.') if s.strip()]
    prompted_sentences = [s.strip() for s in prompted_text.split('.') if s.strip()]

    results = []
    
    for i, (orig, trans) in enumerate(zip(unguided_sentences, prompted_sentences)):
        # find Similarity
        orig_w = set(orig.lower().split())
        trans_w = set(prompted_sentences[i].lower().split())
        
        # find overlap between basic transciption and prompted transcription
        intersection = orig_w.intersection(trans_w)
        similarity = (len(intersection) / len(orig_w)) * 100 if orig_w else 0
        
        results.append({
            'sentence_id': i + 1,
            'condition': 'prompted',
            'original': orig, # meaning basic text
            'transcription': trans, # prmopted transcription
            'word_accuracy': round(similarity, 2),
            'exact_match': (orig.lower() == trans.lower())
        })
    
    return results

# ===========================================
# Step 6: Implementing Audio Chunking
# ===========================================

#split audio into segments using ffmpeg.
def create_audio_chunks(input_path, output_dir, minutes=0.5):
    audio = AudioSegment.from_file(input_path)
    print(f"Total audio duration is {len(audio) / 1000:.2f} seconds.")
    
    chunk_length_ms = int(minutes * 60 * 1000)
    
    # create the output directory to save chunks
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    counter = 1
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i : i + chunk_length_ms]
        filename = output_path / f"chunk_{counter:03d}.mp3"
        chunk.export(filename, format="mp3")
        print(f"Exported: {filename}") # export chunks in a file
        counter += 1

# ======================================================
# Step 7: Transcribing Chunks with Timestamps
# ======================================================

def transcribe_chunks_with_timestamps(chunk_dir, offset_sec=30):
    chunk_files = sorted(list(Path(chunk_dir).glob("*.mp3")))
    full_transcript = []
    current_offset = 0
    for chunk_path in chunk_files:
        print(f"Transcribing {chunk_path.name}...")
        with open(chunk_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="verbose_json" # to get timestamp data
            )
        
        for segment in transcript.segments:
            adjusted_start = segment.start + current_offset
            adjusted_end = segment.end + current_offset
        
            full_transcript.append({
                "start": round(adjusted_start, 2),
                "end": round(adjusted_end, 2),
                "text": segment.text.strip()
            })
            
        # Increase the offset for the next chunk
        current_offset += offset_sec
        
    return full_transcript
    
# ======================================================
# Testing Block: Execution goes HERE
# ======================================================
if __name__ == "__main__":
    print("Running original transcription script...")
    
    # Step 2
    verify_audio_file(audio_path)
    # Step 3
    transcription_text = transcribe_audio_file_as_is(audio_path)
    print("\n📝 Transcription:\n", "-" * 40, "\n", transcription_text.text, "\n", "-" * 40)
    
    # Step 4
    result = transcribe_audio_file_with_prompt(audio_path, prompt_for_transcription)
    print("--- Prompted Transcription ---\n", result.text)
    
    # Step 5
    result_analysis = analyse_transcripts()
    df_results = pd.DataFrame(result_analysis)
    for sentence_id in range(1, len(result_analysis) + 1):
        row = df_results[df_results['sentence_id'] == sentence_id].iloc[0]
        status = "✅" if row['word_accuracy'] >= 99.0 else "⚠️"
        print(f"\n### Sentence {sentence_id}")
        print(f"{status} Comparison (Unguided vs Prompted):")
        print(f"   Unprompted: \"{row['original']}\"")
        print(f"   Prompted: \"{row['transcription']}\"")
        print(f"   Similarity: {row['word_accuracy']}%")
    
    # Step 6
    create_audio_chunks(audio_path, AUDIO_DATA_DIR / "chunks", 0.5)
    
    # Step 7 & 8
    final_data = transcribe_chunks_with_timestamps(AUDIO_DATA_DIR / "chunks", 30)

    
    with open("final_transcript.json", "w") as f:
        json.dump(final_data, f, indent=4)
        
    # export to txt file
    with open("final_transcript.txt", "w") as f:
        for item in final_data:
            f.write(f"[{item['start']}s - {item['end']}s]: {item['text']}\n")
    
    # export to SRT (Subtitles)
    with open("final_transcript.srt", "w") as f:
        for i, item in enumerate(final_data, 1):
            f.write(f"{i}\n{item['start']} --> {item['end']}\n{item['text']}\n\n")
            
    print("✅ Exported: final_transcript.txt, final_transcript.srt and json")