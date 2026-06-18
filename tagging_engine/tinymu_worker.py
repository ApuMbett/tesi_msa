"""tinymu_worker.py"""
import warnings
import torch
import sys
from pathlib import Path

# Add repo to path
_HERE = Path(__file__).parent.resolve()
_TINYMU = _HERE / "TinyMU"
sys.path.insert(0, str(_TINYMU / "src"))

from train_accelerate import get_model_and_tokenizer
from base_worker import BaseWorker

import contextlib
warnings.filterwarnings("ignore")

class TinyMUWorker(BaseWorker):
    def load_model(self):
        print("[worker] Loading TinyMU (229M)...", file=sys.stderr, flush=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with contextlib.redirect_stdout(sys.stderr):  # suppress noisy library logs
            self.model, self.tokenizer, _ = get_model_and_tokenizer(
                config= str(_TINYMU / "ckpt/tinymu.yaml"),
                model_ckpt_path= str(_TINYMU / "ckpt/tinymu.pt")
            )
        self.model.to(self.device)
        self.model.eval()

    def process_audio(self, audio_path: str) -> list[str]:
        prompt = "What instruments are used in this music? List them concisely."
        with torch.no_grad():
            response = self.model.generate(
                samples=[(audio_path, prompt)],
                max_len=50,
                top_p=0.5,
                temperature=0.3,
                tokenizer=self.tokenizer,
                strategy="top-p",
                device=self.device
            )
        
        raw_text = response[0] if isinstance(response, list) else str(response)
        # Parse logic here if needed, or just return as a 1-item list
        return [raw_text.strip()]

if __name__ == "__main__":
    worker = TinyMUWorker()
    worker.run()