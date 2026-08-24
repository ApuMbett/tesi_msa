import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrackWorkspace:
    """
    Manages the filesystem layout for a single track's assets.
    Eliminates scattered outputs by grouping stems, tokens, caches, and final JSONs
    under a unified dataset root.
    """
    #TODO Ideally here we want to save also the data that composes track_identity so basically author and song name. 
    track_name: str
    original_audio_path: Optional[Path] = None

    def __post_init__(self):
        # Resolve the Dataset Root from the environment.
        # Fall back to the default project-local 'data/processed' directory for convenience.
        dataset_root_env = os.getenv("SEMANTIC_MSA_DATASET_ROOT")
        if dataset_root_env:
            self.root = Path(dataset_root_env).resolve()
        else:
            # Fallback relative to the repository root (tesi)
            repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            self.root = repo_root / "data" / "processed"

        self.track_dir = self.root / self.track_name
        self.audio_dir = self.track_dir / "audio"
        self.stems_dir = self.track_dir / "stems"
        self.tokens_dir = self.track_dir / "tokens"
        self.cache_dir = self.track_dir / "cache"

        # The final full.json and LLM.json will sit at the root of the track_dir
        self.output_dir = self.track_dir

    def setup(self) -> Path:
        """
        Creates the workspace directories and copies the source audio into it.
        Returns the path to the internal source audio.
        """
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.stems_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.original_audio_path:
            # Assuming it might already be in the workspace
            internal_audio = self.audio_dir / "source.mp3"
            if not internal_audio.exists():
                raise FileNotFoundError(f"Source audio not found in workspace, and no original path provided: {internal_audio}")
            return internal_audio

        internal_audio = self.audio_dir / self.original_audio_path.name
        
        # Avoid copying if it's the exact same file path
        if self.original_audio_path.resolve() != internal_audio.resolve():
            if not internal_audio.exists() or os.path.getsize(self.original_audio_path) != os.path.getsize(internal_audio):
                shutil.copy2(self.original_audio_path, internal_audio)
                
        return internal_audio

    @property
    def source_audio_path(self) -> Path:
        """Returns the expected path of the audio source inside the workspace."""
        if self.original_audio_path:
            return self.audio_dir / self.original_audio_path.name
        return self.audio_dir / "source.mp3"

