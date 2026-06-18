"""
qwen_worker.py 
==============
Hybrid worker: Routes requests to Hugging Face Serverless API (0 VRAM)
or runs Qwen2-Audio-7B locally (~16GB VRAM) based on environment variables.
"""

import os
import sys
import json
import base64
import warnings
from base_worker import BaseWorker

warnings.filterwarnings("ignore")

class QwenWorker(BaseWorker):
    def load_model(self):
        # 1. The Toggle: Check environment variable (defaults to Local if not set)
        self.use_api = os.environ.get("USE_QWEN_API", "false").lower() == "true"
        
        self.prompt = """Listen to this audio. Identify the top 3 dominant musical instruments or synthesizers. 
        Output ONLY a valid JSON array of strings."""
        
        if self.use_api:
            # ---------------------------------------------------------
            # API MODE (Lightweight, 0 VRAM)
            # ---------------------------------------------------------
            print("[worker] Initializing Qwen via Hugging Face API...", file=sys.stderr, flush=True)
            self.api_token = os.environ.get("HF_TOKEN")
            if not self.api_token:
                raise ValueError("HF_TOKEN environment variable is required to use API mode.")
            
            # Use the official InferenceClient
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_token)
        else:
            # ---------------------------------------------------------
            # LOCAL MODE (Heavy, ~16GB VRAM)
            # ---------------------------------------------------------
            print("[worker] Loading Qwen2-Audio-7B Locally...", file=sys.stderr, flush=True)
            import torch
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct", 
                device_map="auto", 
                torch_dtype=torch.bfloat16
            )
            self.model.eval()

    def process_audio(self, audio_path: str) -> list[str]:
        # Route the IPC payload to the correct execution engine
        if self.use_api:
            return self._process_api(audio_path)
        else:
            return self._process_local(audio_path)

    def _process_api(self, audio_path: str) -> list[str]:
        """Executes inference via the free HF Serverless API."""
        # Read the .wav file and encode to base64 for network transit
        with open(audio_path, "rb") as f:
            audio_data = f.read()
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        
        # Standard OpenAI-compatible multimodal message format
        messages = [
            {"role": "system", "content": "You are a strict JSON data extractor."},
            {"role": "user", "content": [
                {"type": "text", "text": self.prompt},
                # HF Inference handles base64 data URIs
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}} 
            ]}
        ]
        
        # Trigger the remote endpoint
        response = self.client.chat.completions.create(
            model="Qwen/Qwen2-Audio-7B-Instruct",
            messages=messages,
            max_tokens=64,
            temperature=0.1
        )
        
        raw_text = response.choices[0].message.content.strip()
        return self._clean_json(raw_text)

    def _process_local(self, audio_path: str) -> list[str]:
        """Executes inference on your local GPU."""
        import librosa
        import torch
        
        messages = [
            {"role": "system", "content": "You are a strict JSON data extractor."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": self.prompt}
            ]}
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audio, _ = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
        
        inputs = self.processor(text=text, audios=[audio], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=64, temperature=0.1, do_sample=False)
        
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        raw_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
        return self._clean_json(raw_text)

    def _clean_json(self, raw_text: str) -> list[str]:
        """Ensures Qwen's output is safe to pass back to the Python 3.9 Orchestrator."""
        clean_json = raw_text.replace("```json", "").replace("```", "").strip()
        import json
        return json.loads(clean_json)

if __name__ == "__main__":
    worker = QwenWorker()
    worker.run()