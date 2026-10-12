from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from semantic_msa.agents.core.base import BaseAgent
from semantic_msa.agents.core.factory import LLMFactory
from semantic_msa.agents.core.schemas import DirectAudioResponseDTO
from semantic_msa.utils.logger import logger


class DirectAudioAgent(BaseAgent):
    """
    End-to-End Direct Audio Baseline Agent.
    Sends raw audio directly to gemini-3.5-flash-lite with metadata (title, artist, duration)
    to perform zero-shot functional Music Structure Analysis (SongFormer 7-class taxonomy).
    Outputs the raw JSON without automated post-processing adjustments.
    """

    def __init__(self, config: Dict[str, Any], factory: LLMFactory):
        cfg = dict(config)
        cfg.setdefault("model", "gemini-3.5-flash-lite")
        cfg.setdefault("temperature", 0.7)  # Aligned with Segmenter temperature
        super().__init__("direct_audio", cfg, factory)

    def get_prompt_template(self) -> str:
        prompts_dir = Path(__file__).resolve().parent / "core" / "prompts"
        prompt_path = prompts_dir / "direct_audio.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Direct audio prompt template not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    def format_system_prompt(self, title: str, artist: str, duration: float) -> str:
        template = self.get_prompt_template()
        return (
            template.replace("{title}", str(title))
            .replace("{artist}", str(artist))
            .replace("{duration}", f"{duration:.2f}")
        )

    def format_prompt(self, title: str, artist: str, duration: float) -> str:
        """Alias for format_system_prompt."""
        return self.format_system_prompt(title=title, artist=artist, duration=duration)

    def run(
        self,
        audio_path: Union[str, Path],
        title: str = "Unknown Track",
        artist: str = "Unknown Artist",
        duration: float = 0.0,
    ) -> Tuple[str, DirectAudioResponseDTO, Dict[str, Any]]:
        """
        Executes end-to-end direct audio inference on the provided audio file.
        Returns:
            (prompt_text, parsed_dto, result_dict)
        """
        system_prompt = self.format_system_prompt(title=title, artist=artist, duration=duration)
        user_message = f'Track: "{title}" by {artist}. Duration: {duration:.2f} seconds.\nPlease analyze the attached audio and provide the structural segmentation.'

        raw_response, parsed_dto = self.generate_and_parse(
            system_prompt=system_prompt,
            user_message=user_message,
            schema_class=DirectAudioResponseDTO,
            audio_path=audio_path,
        )

        segments = [s.model_dump() for s in parsed_dto.segments]

        result_dict = {
            "segments": segments,
            "metadata": {
                "title": title,
                "artist": artist,
                "duration": duration,
                "model": self.model,
                "temperature": self.temperature,
            },
        }

        return system_prompt, parsed_dto, result_dict
