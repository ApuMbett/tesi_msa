import sys
import json
from BeatNet.BeatNet import BeatNet
from pathlib import Path

def process_track(audio_path: str, output_path: str):
    """
    Runs BeatNet on the audio and outputs the beat grid to the JSON file.
    Output format: List of [time_ms, beat_number]
    """
    if not Path(audio_path).exists():
        print(f"Error: audio file {audio_path} does not exist.")
        sys.exit(1)

    print(f"Extracting beat grid for {audio_path}...", file=sys.stderr)
    estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
    raw_data = estimator.process(audio_path)
    
    # Sanitize: Convert to (time_ms, beat_index)
    sanitized = [[int(time * 1000), int(beat)] for time, beat in raw_data]
    
    with open(output_path, "w") as f:
        json.dump(sanitized, f)
        
    print(f"Successfully wrote beat grid to {output_path}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python beatnet_wrapper.py <audio_path> <output_json_path>")
        sys.exit(1)
        
    process_track(sys.argv[1], sys.argv[2])
