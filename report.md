# Speech-to-Text Implementation with Whisper

## Differences Between Prompted and Unprompted Transcription
The implementation utilized OpenAI’s Whisper API to process audio files under two conditions: unprompted (unguided) and prompted (guided).

The pipeline runs the same audio through Whisper twice: once with no guidance (`transcribe_audio_file_as_is`) and once with a prompt (`transcribe_audio_file_with_prompt`) listing expected terms — "moon landing, Maine, soil sample, kangaroo walk." The prompt nudges Whisper API toward those words and their surrounding phonetic patterns when the audio is ambiguous, without forcing them into the output. In practice, this mainly helps with:

- **Unprompted (Unguided):** The model relies entirely on its pre-trained knowledge. While highly accurate, it is susceptible to misinterpreting context-specific terminology, proper nouns, or ambiguous phrasing (e.g., "kangaroo walk" vs. "what do they call it").

- **Prompted (Guided):** By providing the model with a "prompt" containing keywords (e.g., "moon landing," "Maine," "soil sample"), the API effectively steers its predictive capabilities. The analysis showed high similarity between conditions, but the prompted version consistently demonstrated superior handling of niche technical terms or specific speech patterns.

It is worth noting that neither transcript is a verified ground truth, so a low similarity score tells the two versions disagree, not which one is correct. A true accuracy comparison would require a human-verified reference transcript and a word error rate (WER) calculation.

## Benefits of Chunking for Long Audio
The project splits the source audio into 30-second segments before transcription. This provides several concrete benefits:
- **API limits:** Whisper's API has a file size cap (`25MB`); chunking keeps each request safely under that regardless of source file length.
- **Timestamp granularity:** Using `response_format="verbose_json"` on each chunk returns segment-level timestamps, which the code reassembles into a continuous timeline by adding a running offset. This makes subtitle (SRT) and timestamped-transcript export possible.

## Challenges Encountered
Several technical hurdles typical of audio processing:
- **The "Copy" Effect:** When using pydub to chunk files, logic errors—such as setting the chunk length larger than the file duration—resulted in the script creating identical copies of the source file rather than segments. Since the original file length is low, so had to set the segment size really low `0.5`
- **Data Type Strictness:** Working with Python’s range() function and OpenAI’s SDK objects triggered `TypeError` exceptions. Specifically, passing float values to range() and attempting to treat TranscriptionSegment objects as dictionaries (using brackets instead of dot notation) required strict type casting and syntax adjustments.

## Recommendations for Improving Accuracy
- For future iteration, I would like to implement proper error handling: Both transcription functions and the chunk loop have no try/except around the API calls, so a single failed chunk (rate limit, timeout, malformed audio) will crash the entire batch rather than being logged and skipped.
- **Audio Pre-processing:** Implementing noise reduction or normalization via pydub before sending audio to the API would improve the signal-to-noise ratio, further increasing transcription quality.
- **Standardized Validation:** Expand the analysis script to use word-level error rates (WER) rather than simple similarity sets to provide more robust verification of transcription accuracy.