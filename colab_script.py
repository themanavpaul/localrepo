#!/usr/bin/env python3

# @title 1. Cleanup (Optional: Clears previous /content/ files)
# ---
!rm -rf /content/*
print("Cleaned up /content/ directory.")

# @title 2. Check GPU Availability
# ---
!nvidia-smi

# @title 3. Install Dependencies into Virtual Environment
# ---
import os

# Install virtualenv
!pip install virtualenv -q

# Create a virtual environment named 'env' in /content/
# The -p python3 option ensures it uses the system's python3
!virtualenv /content/env -p python3
print("Virtual environment 'env' created.")

# Install ctranslate2 (pinning as per original notebook, whisperx might have its own preference)
!/content/env/bin/pip install ctranslate2==4.4.0 -q
print("Installed ctranslate2 into virtual environment.")

# Install whisperx (this will pull compatible torch, torchaudio, etc.)
!/content/env/bin/pip install git+https://github.com/m-bain/whisperx.git -q
print("Installed whisperx and its dependencies (like torch, torchaudio) into virtual environment.")

# Install ffmpeg (system-wide, usually available in Colab but good to ensure)
!apt-get update -qq
!apt-get install -y ffmpeg -qq
print("ffmpeg installed/updated.")

# @title 4. Upload Video File
# ---
from google.colab import files
import os

print("Please upload your video file:")
uploaded = files.upload()

video_file_path = None
audio_file_path = None # Will be derived from video_file_path

if uploaded:
    video_file_name = next(iter(uploaded))
    
    # Ensure the uploaded file is in /content/
    # files.upload() typically saves to the current working directory, which is /content/ in Colab
    source_path = video_file_name 
    destination_path = os.path.join("/content", os.path.basename(video_file_name))

    if source_path != destination_path:
        os.rename(source_path, destination_path)
        print(f"Moved '{source_path}' to '{destination_path}'")
    else:
        print(f"File '{destination_path}' is already in /content/")

    video_file_path = destination_path
    print(f"Video file path: {video_file_path}")

    # Define audio file path based on video name
    base_name, _ = os.path.splitext(os.path.basename(video_file_path))
    audio_file_path = os.path.join("/content", base_name + ".wav")
    print(f"Target audio file path: {audio_file_path}")
else:
    print("No file uploaded. Please upload a video file to proceed.")

# @title 5. Extract Audio from Video
# ---
import os

if video_file_path and os.path.exists(video_file_path):
    # Ensure audio_file_path is defined (it should be from the previous cell)
    if audio_file_path:
        # Extract audio using ffmpeg: 16000 Hz sample rate, mono, 16-bit PCM WAV
        ffmpeg_command = f"ffmpeg -y -i \"{video_file_path}\" -ar 16000 -ac 1 -c:a pcm_s16le \"{audio_file_path}\""
        print(f"Executing: {ffmpeg_command}")
        
        exit_code = os.system(ffmpeg_command)
        
        if exit_code == 0 and os.path.exists(audio_file_path):
            print(f"Audio extracted successfully to {audio_file_path}")
        elif exit_code == 0 and not os.path.exists(audio_file_path):
            print(f"ERROR: ffmpeg reported success (exit code 0) but {audio_file_path} was not found! Check ffmpeg output.")
            audio_file_path = None # Indicate failure
        else:
            print(f"Error extracting audio. ffmpeg exit code: {exit_code}. Check ffmpeg output above for details.")
            audio_file_path = None # Indicate failure
    else:
        print("Audio file path not defined. Cannot extract audio.")
        audio_file_path = None
elif not video_file_path:
    print("Video file not uploaded. Skipping audio extraction.")
else: # video_file_path is defined but file doesn't exist
    print(f"Video file {video_file_path} not found. Skipping audio extraction.")


# @title 6. Transcribe Audio and Generate Subtitles (Direct Execution)
# ---
import os

subtitles_ass_path = "/content/subtitles.ass" 

# Remove old subtitles file if it exists, to ensure the script creates a new one
if os.path.exists(subtitles_ass_path):
    os.remove(subtitles_ass_path)
    print(f"Removed old subtitles file: {subtitles_ass_path}")

if audio_file_path and os.path.exists(audio_file_path):
    # Prepare the audio_file_path to be embedded as a Python string literal in the script
    # repr() ensures it's correctly quoted and escaped.
    python_string_audio_path = repr(str(audio_file_path))

    # Python script content
    # AUDIO_FILE_PATH is directly embedded into this script string.
    run_transcription_code = f"""
import whisperx
import torch
import gc # For garbage collection
from datetime import timedelta
import os
import sys # For sys.exit

# --- Configuration ---
AUDIO_FILE_PATH = {python_string_audio_path} # Path is injected here

MODEL_SIZE = "medium" # Options: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
COMPUTE_TYPE = "float16" # Options: "float16", "int8_float16", "int8" (for faster-whisper)
BATCH_SIZE = 16 # Reduce if Out Of Memory (OOM)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBTITLES_OUTPUT_PATH = "/content/subtitles.ass"

# --- Helper Functions ---
def format_timestamp_ass(seconds):
    if seconds is None: return "0:00:00,00" # Default for safety
    try:
        float_seconds = float(seconds)
    except (ValueError, TypeError):
        print(f"Warning: Invalid seconds value for timestamp: {{seconds}}. Using 0.")
        float_seconds = 0.0
    td = timedelta(seconds=float_seconds)
    total_seconds_int = int(td.total_seconds())
    hours = total_seconds_int // 3600
    minutes = (total_seconds_int % 3600) // 60
    secs = total_seconds_int % 60
    centiseconds = int(td.microseconds / 10000)
    return f"{{hours}}:{{minutes:02}}:{{secs:02}},{{centiseconds:02}}" # Inner f-string, so double braces

def generate_ass_content(aligned_transcription_data):
    ass_content = "[Script Info]\\n"
    ass_content += "Title: Transcription via WhisperX\\n"
    ass_content += "ScriptType: v4.00+\\n"
    ass_content += "WrapStyle: 0\\n"
    ass_content += "ScaledBorderAndShadow: yes\\n"
    ass_content += "YCbCr Matrix: None\\n\\n"
    ass_content += "[V4+ Styles]\\n"
    ass_content += "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\\n"
    ass_content += "Style: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,20,1\\n\\n"
    ass_content += "[Events]\\n"
    ass_content += "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\\n"

    if aligned_transcription_data and "segments" in aligned_transcription_data:
        for segment in aligned_transcription_data["segments"]:
            if "words" in segment and segment["words"] and all("start" in word and "end" in word and "word" in word for word in segment["words"]):
                for word_info in segment["words"]:
                    start_time = word_info.get("start")
                    end_time = word_info.get("end")
                    text = word_info.get("word", "").strip().replace("\\n", " ")
                    if text and start_time is not None and end_time is not None:
                        start_ts = format_timestamp_ass(start_time)
                        end_ts = format_timestamp_ass(end_time)
                        ass_content += f"Dialogue: 0,{{start_ts}},{{end_ts}},Default,,0,0,0,,{{text}}\\n" # Inner f-string
            elif "start" in segment and "end" in segment and "text" in segment:
                start_time = segment.get("start")
                end_time = segment.get("end")
                text = segment.get("text", "").strip().replace("\\n", " ")
                if text and start_time is not None and end_time is not None:
                    start_ts = format_timestamp_ass(start_time)
                    end_ts = format_timestamp_ass(end_time)
                    ass_content += f"Dialogue: 0,{{start_ts}},{{end_ts}},Default,,0,0,0,,{{text}}\\n" # Inner f-string
    else:
        print("Warning: No segments found in aligned transcription data for ASS generation.")
    return ass_content

# --- Main Transcription Logic ---
model = None
model_a = None
aligned_result = None

try:
    if AUDIO_FILE_PATH == 'None' or not os.path.exists(AUDIO_FILE_PATH): # Check if path is valid (repr('None') is "'None'")
        print(f"Error: Audio file not found at '{{AUDIO_FILE_PATH}}' or path is invalid.")
        sys.exit(1)

    print(f"Script starting. Processing audio: {{AUDIO_FILE_PATH}}") # Inner f-string
    print(f"Using Device: {{DEVICE}}") # Inner f-string
    print(f"Loading Whisper model: {{MODEL_SIZE}} (compute type: {{COMPUTE_TYPE}})") # Inner f-string
    
    try:
        model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"Error loading whisper model: {{e}}") # Inner f-string
        sys.exit(1) # Exits script, outer finally will run

    print(f"Loading audio from: {{AUDIO_FILE_PATH}}") # Inner f-string
    try:
        audio = whisperx.load_audio(AUDIO_FILE_PATH)
    except Exception as e:
        print(f"Error loading audio file: {{e}}") # Inner f-string
        sys.exit(1) # Exits script, outer finally will run
    
    print("Transcribing audio...")
    try:
        transcription_result = model.transcribe(audio, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Error during transcription: {{e}}") # Inner f-string
        sys.exit(1) # Exits script, outer finally will run

    detected_language = transcription_result.get("language")
    print(f"Detected language: {{detected_language}}") # Inner f-string

    if not detected_language:
        print("Error: Language not detected by Whisper. Cannot proceed with alignment.")
        sys.exit(1) # Exits script, outer finally will run
        
    print("Loading alignment model for language: {{detected_language}}...") # Inner f-string
    try:
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=DEVICE)
        print("Aligning transcription...")
        if "segments" in transcription_result and transcription_result["segments"]:
             aligned_result = whisperx.align(transcription_result["segments"], model_a, metadata, audio, DEVICE)
        else:
            print("Warning: No segments found in transcription result to align.")
            aligned_result = {{ "segments": [] }} # Provide empty structure
    except Exception as e:
        print(f"Error during alignment: {{e}}") # Inner f-string
        # aligned_result will remain None or its previous state
    finally:
        if model_a:
            del model_a
            model_a = None # Explicitly set to None after deletion
            print("Alignment model cleaned up.")
        # Main model 'model' is cleaned in the outermost finally block

    if aligned_result:
        print("Transcription and alignment complete.")
        ass_file_content = generate_ass_content(aligned_result)
        try:
            with open(SUBTITLES_OUTPUT_PATH, "w", encoding="utf-8") as f:
                f.write(ass_file_content)
            print(f"Subtitles written to {{SUBTITLES_OUTPUT_PATH}}") # Inner f-string
        except Exception as e:
            print(f"Error writing .ass file: {{e}}") # Inner f-string
            sys.exit(1) # Exits script, outer finally will run
    else:
        print("Alignment step did not produce a result or an error occurred.")
        sys.exit(1) # Exits script, outer finally will run
        
    print("Script finished successfully.")

finally:
    # This outermost finally block ensures cleanup happens even if sys.exit() was called.
    print("Performing final cleanup of models and resources...")
    if model:
        del model
        model = None
        print("Main whisper model cleaned up.")
    if model_a: # Should be None if inner finally ran, but as a safeguard
        del model_a
        model_a = None
        print("Alignment model cleaned up (final check).")
    if DEVICE == "cuda":
        print("Emptying CUDA cache...")
        torch.cuda.empty_cache()
    print("Running garbage collection...")
    gc.collect()
    print("Final cleanup complete.")
"""

    # To run this code directly, it's more robust to write it to a temporary file
    # and execute that file, rather than dealing with complex shell escapes for `python -c`.
    temp_script_path = "/content/_temp_whisperx_run_script.py"
    try:
        with open(temp_script_path, "w", encoding="utf-8") as f:
            f.write(run_transcription_code)
        print(f"Temporary script for transcription created at {temp_script_path}")

        execution_command = f"/content/env/bin/python {temp_script_path}"
        print(f"Executing transcription script: {execution_command}")
        
        # os.system will show output directly in the Colab cell
        exit_code_script = os.system(execution_command)

        if exit_code_script == 0 and os.path.exists(subtitles_ass_path):
            print(f"Transcription script finished successfully. Subtitles at {subtitles_ass_path}")
        elif exit_code_script == 0 and not os.path.exists(subtitles_ass_path):
            print(f"Transcription script finished (exit code 0) but subtitles file {subtitles_ass_path} was NOT created. Check script output above.")
        else:
            print(f"Transcription script failed with exit code {exit_code_script}. Check script output above for errors.")
            
    except Exception as e:
        print(f"An error occurred in the Colab cell while preparing or running the script: {e}")
    finally:
        # Clean up the temporary script file
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
            print(f"Removed temporary script: {temp_script_path}")
else:
    if not audio_file_path:
        print("Skipping transcription: Audio file path was not set (e.g., no video uploaded or audio extraction failed).")
    elif not os.path.exists(audio_file_path):
         print(f"Skipping transcription: Audio file '{str(audio_file_path)}' not found.")


# @title 7. (This step is now merged into Step 6)
# ---
# This cell is intentionally left blank as its functionality was merged into Step 6.
print("Step 7 was merged into Step 6.")
# ---

# @title 8. Burn Subtitles into Video
# ---
import os

# video_file_path should be defined from Cell 4
# subtitles_ass_path is defined as "/content/subtitles.ass"
output_video_with_subs_path = None # Initialize

if video_file_path and os.path.exists(video_file_path):
    if os.path.exists(subtitles_ass_path):
        base_name_video, ext_video = os.path.splitext(os.path.basename(video_file_path))
        output_video_with_subs_path = os.path.join("/content", base_name_video + "_subtitled" + ext_video)

        # Burn subtitles into the video using ffmpeg. -y overwrites output.
        ffmpeg_burn_command = f"ffmpeg -y -i \"{video_file_path}\" -vf ass=\"{subtitles_ass_path}\" -c:a copy \"{output_video_with_subs_path}\""
        
        print(f"Executing subtitle burn command: {ffmpeg_burn_command}")
        exit_code_burn = os.system(ffmpeg_burn_command)
        
        if exit_code_burn == 0 and os.path.exists(output_video_with_subs_path):
            print(f"Subtitles burned successfully. Output video: {output_video_with_subs_path}")
        elif exit_code_burn == 0 and not os.path.exists(output_video_with_subs_path):
             print(f"ERROR: ffmpeg reported success for burning subs, but output file {output_video_with_subs_path} not found!")
             output_video_with_subs_path = None
        else:
            print(f"Error burning subtitles. ffmpeg exit code: {exit_code_burn}. Check ffmpeg output.")
            output_video_with_subs_path = None
    else:
        print(f"Subtitles file '{subtitles_ass_path}' not found. Skipping subtitle burning.")
elif not video_file_path:
     print("Original video file not available. Skipping subtitle burning.")
else: # video_file_path defined but doesn't exist
    print(f"Original video file '{video_file_path}' not found. Skipping subtitle burning.")


# @title 9. Download Video with Subtitles
# ---
from google.colab import files

if output_video_with_subs_path and os.path.exists(output_video_with_subs_path):
    print(f"Preparing '{os.path.basename(output_video_with_subs_path)}' for download...")
    files.download(output_video_with_subs_path)
else:
    print("Output video with subtitles is not available for download. Check previous steps for errors.")
