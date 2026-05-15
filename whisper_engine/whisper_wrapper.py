import sys
import os
import json
import torch
import whisperx

# ==========================================
# We will define the custom SyedRMSVad class here later!
# ==========================================

if __name__ == "__main__":
    # 1. Catch the audio file path passed by the DSPManager
    if len(sys.argv) < 2:
        print("Error: No audio file provided.")
        print("Usage: python run_custom_whisper.py <path_to_vocals.wav>")
        sys.exit(1)
        
    vocals_path = sys.argv[1]
    
    if not os.path.exists(vocals_path):
        print(f"Error: Could not find audio file at {vocals_path}")
        sys.exit(1)

    # 2. Set up Device & Compute Type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # 3. Load WhisperX Model
    print(f"Loading WhisperX (large-v2) on {device}...")
    whisper_model = whisperx.load_model(
        "large-v2", 
        device, 
        compute_type=compute_type
        # vad_model=my_custom_vad  <-- We will inject Syed's VAD here later!
    )
    print("Model loaded successfully.")

    # 4. Load Audio
    print(f"Loading audio: {vocals_path}")
    audio = whisperx.load_audio(vocals_path)
    
    # 5. Base Transcription
    print("Running transcription...")
    result = whisper_model.transcribe(audio, batch_size=16)
    
    # 6. Forced Alignment (Word-level timestamps)
    print("Running forced alignment for DJ grid precision...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=device
    )
    
    aligned_result = whisperx.align(
        result["segments"], 
        align_model, 
        metadata, 
        audio, 
        device, 
        return_char_alignments=False
    )
    
    # 7. Write the output to a JSON file for the main thesis environment to read
    track_name = vocals_path.split("/")[-2]  
    # create the output directory if it doesn't exist
    os.makedirs(f"../data/json_db/{track_name}", exist_ok=True)
    output_filename = f"../data/json_db/{track_name}/_whisper_output.json"
    print(f"Saving exact timestamps to {output_filename}...")
    print(json.dumps(aligned_result, indent=2))  # Debug: print the aligned result to console
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(aligned_result, f, ensure_ascii=False, indent=2)
        
    print("WhisperX Bridge Script completed successfully!")