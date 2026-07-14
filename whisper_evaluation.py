import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import time
from pathlib import Path
from whisper_transcription import transcribe_audio_file_as_is, transcribe_audio_file_with_prompt
 
def record_audio(duration=30, sample_rate=16000):
    """Record audio for specified duration"""
    print(f"🎤 Recording for {duration} seconds...")
    print("Get ready...")
    time.sleep(2)
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("🔴 RECORDING! Speak now!")
    
    # Record audio
    audio = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, 
                   channels=1, 
                   dtype='float32')
    sd.wait()
    
    print("✅ Recording complete!")
    return audio.flatten(), sample_rate
 
# Record your audio
audio, sr = record_audio(duration=60)  # 45 seconds
 
# Save to file
audio = np.clip(audio, -1, 1)
audio_int16 = (audio * 32767).astype(np.int16)
recording_path = Path("audio/akansha_recording.wav")
wavfile.write(recording_path, sr, audio_int16)
print("✅ Saved to 'akansha_recording.wav'")

print("Starting transcription using code from whisper_transcription.py...")

result = transcribe_audio_file_as_is(audio=recording_path)

print("Transcription complete!\n", result.text)

