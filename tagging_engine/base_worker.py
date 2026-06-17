"""
base_worker.py
==============
The abstract IPC daemon. It handles the JSON pipe loop so your specific 
model scripts don't have to rewrite the boilerplate.
"""

import sys
import json
import traceback

class BaseWorker:
    def __init__(self):
        # Allow the subclass to load its specific model weights into VRAM
        self.load_model()
        
        # Once the subclass finishes loading, send the ready handshake
        self.send_response({"status": "ready"})
        print(f"[worker] {self.__class__.__name__} ready", file=sys.stderr, flush=True)

    def send_response(self, obj: dict | list):
        print(json.dumps(obj), flush=True)

    def run(self):
        """The main IPC daemon loop."""
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                request = json.loads(raw_line)
                audio_path = request["path"]
            except (json.JSONDecodeError, KeyError) as exc:
                self.send_response({"error": f"Invalid request format: {str(exc)}"})
                continue

            try:
                # Ask the subclass to process the audio
                tags = self.process_audio(audio_path)
                self.send_response(tags)
            except Exception as exc:
                # Catch hallucinations or inference crashes cleanly
                print(f"[worker error] {traceback.format_exc()}", file=sys.stderr, flush=True)
                self.send_response(["Unknown Timbre"])

    # -----------------------------------------------------------------
    # Methods to be overridden by subclasses (Qwen, TinyMU, FineLAP)
    # -----------------------------------------------------------------
    def load_model(self):
        raise NotImplementedError("Subclasses must implement load_model()")

    def process_audio(self, audio_path: str) -> list[str]:
        raise NotImplementedError("Subclasses must implement process_audio()")