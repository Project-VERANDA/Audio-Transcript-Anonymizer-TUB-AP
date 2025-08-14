import os
import time
import ffmpeg
import InputToTranscript
import ner_deidentify_evaluation as RedactReplace

# Define paths
base_path = "/Users/hamann/Documents/Uni/SoSe25/QU Project/group_github/anonymize-medical-data/anonymize-medical-data" # *** change main path to the location of this file (get current path by typing 'pwd' into your terminal)
videos_folder = os.path.join(base_path, "videos")
audios_folder = os.path.join(base_path, "audios")
transcripts_folder = os.path.join(base_path, "transcripts")
annonym_folder = os.path.join(base_path, "annonym")
test_data_txt = os.path.join(base_path, "data/test_data")

# Create transcripts folder if doesn't exist
if not os.path.exists(transcripts_folder):
    os.makedirs(transcripts_folder)

# Create annonym folder if doesn't exist
if not os.path.exists(annonym_folder):
    os.makedirs(annonym_folder)


##############################################
### CREATE TRANSCRIPTS FROM VIDEO OR AUDIO ###
##############################################

# 1. Check if videos folder exists, extract audios if needed
if os.path.exists(videos_folder):
    print("Found 'videos' folder. Extracting audio from MP4 files...")

    if not os.path.exists(audios_folder):
        os.makedirs(audios_folder)

    for file in os.listdir(videos_folder):
        if file.endswith(".mp4"):
            video_path = os.path.join(videos_folder, file)
            audio_filename = os.path.splitext(file)[0] + ".mp3"
            audio_path = os.path.join(audios_folder, audio_filename)

            if not os.path.exists(audio_path):
                print(f"Extracting audio from {video_path}...")
                ffmpeg.input(video_path).output(audio_path, format="mp3", audio_bitrate="192k").run(overwrite_output=True)
                print(f"Saved extracted audio: {audio_path}")
            else:
                print(f"Audio already extracted: {audio_path}")

else:
    print("No 'videos' folder found. Skipping video extraction.")


# 2. Transcribe audio
if os.path.exists(audios_folder):
    print("Found 'audios' folder. Starting transcription process...")

    # InputToTranscript.transcript_from_audio()
    
    #######################################
    ###     DE-IDENTIFY TRANSCRIPTS     ###
    #######################################
    # when using the whole pipline with whisperx, use 'transcripts_folder' instead of 'test_data_txt'
    for file in os.listdir(test_data_txt):
        file_path = os.path.join(test_data_txt, file)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        tic = time.perf_counter()   # start timer
        deidentified_doc = RedactReplace.deidentify_entities_in_doc(file, text, redact_replace="REDACT")  # Use REDACT or REDACT_STAR as needed
        toc = time.perf_counter()  # end timer

        # Save the anonymized text to a new file
        anonymized_file_path = os.path.join(annonym_folder, os.path.splitext(file)[0] + "_anonymized.txt")
        with open(anonymized_file_path, 'w', encoding='utf-8') as f:
            f.write(str(deidentified_doc))
        print(f"Anonymized transcript saved: {anonymized_file_path}")

    print(f"De-identification completed in {toc - tic:0.4f} seconds.")
else:
    print("No 'audios' folder found. No files to transcribe.")