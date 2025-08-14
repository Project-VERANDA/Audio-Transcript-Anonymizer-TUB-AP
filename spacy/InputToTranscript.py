import os
import whisperx

base_path = "/Users/hamann/Documents/Uni/SoSe25/QU Project/group_github/anonymize-medical-data/anonymize-medical-data" # *** change main path to the location of this file (get current path by typing 'pwd' into your terminal)
videos_folder = os.path.join(base_path, "videos")
audios_folder = os.path.join(base_path, "audios")
transcripts_folder = os.path.join(base_path, "transcripts")
model_folder = os.path.join(base_path, "model")
annonym_folder = os.path.join(base_path, "annonym")
config_file = "philter-config.json"

# Function to merge all consecutive segments of one speaker to one coherent text block
def merge_consecutive_speaker_segments(segments):
    merged_segments = []
    prev_segment = None

    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"]

        if prev_segment and prev_segment["speaker"] == speaker:
            prev_segment["text"] += " " + text
        else:
            if prev_segment:
                merged_segments.append(prev_segment)
            prev_segment = {"speaker": speaker, "text": text}

    if prev_segment:
        merged_segments.append(prev_segment)

    return merged_segments

def transcript_from_audio():
    # *** adjust to your device features (or features of the VM)
    device = "cpu"  # options: "cpu", "cuda" (if GPU available)
    batch_size = 32  # reduce if low on GPU mem
    compute_type = "float32" # change to "int8" if low on GPU mem (may reduce accuracy)

    # Load WhisperX model
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_folder)

    # *** min & max number of speakers in the audio
    min_speakers = 2
    max_speakers = 4

    for file in os.listdir(audios_folder):
        if file.endswith(".mp3"):
            audio_path = os.path.join(audios_folder, file)
            print(f"Processing: {audio_path}")

            # Load and transcribe audio
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=batch_size)
            print(result["segments"])  # Before alignment

            # delete model if low on GPU resources
            # import gc; gc.collect(); torch.cuda.empty_cache(); del model

            # Align Whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            print(result["segments"])  # After alignment

            # delete model if low on GPU resources
            # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

            # Diarization & assign speaker labels
            diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="xxx", device=device)
            diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers) #add min/max number of speakers if known
            
            result = whisperx.assign_word_speakers(diarize_segments, result) # segments are now assigned speaker IDs

            # Merge consecutive speaker segments
            result["segments"] = merge_consecutive_speaker_segments(result["segments"])

            # Save transcript
            transcript_file = os.path.join(transcripts_folder, os.path.splitext(file)[0] + ".txt")
            with open(transcript_file, "w", encoding="utf-8") as f:
                for segment in result["segments"]:
                    speaker = segment.get("speaker", "Unknown")
                    text = segment["text"]
                    f.write(f"{speaker}: {text}\n")

            print(f"Transcript saved to: {transcript_file}")
        
    return 0