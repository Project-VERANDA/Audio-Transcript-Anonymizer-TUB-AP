# Audio-Transcript-Anonymizer

This Pipeline accepts an audio or video file, transcribes the content using WhisperX and applies speaker diarization via Pyannote.
It can be used for interviews, therapy sessions or conversations involving multiple speakers in general.


# Features

Audio/video (mp3/mp4) input

Automatic transcription via WhisperX

Speaker diarization via Pyannote.audio


# Installation

To run the transcription pipeline you'll need Python 3.10. and Anaconda.

1. Installing FFmpeg
   
  Option 1: Via pip
  
    pip install python-ffmpeg
   
  Option 2: Via scoop
  
    scoop install ffmpeg
   
2. Installing WhisperX
   
   Follow the instruction from the WhisperX repository (see 'Setup'):
   
   https://github.com/m-bain/whisperX?tab=readme-ov-file
   
4. Installing Pyannote
   
  Via Pip:

    pip install pyannote.audio
    

# Running the script
Place the script into a folder along with the subfolders 'audios' (for mp3) and/or 'videos' (for mp4) and add your media to the respective folder.

Open the script and update all fields marked with *** and save your changes. 
Run the script. 


