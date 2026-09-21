from typing import Any, Dict, List, Tuple
from semantic_msa.agents.core.base import BaseAgent
from semantic_msa.agents.core.factory import LLMFactory
from semantic_msa.agents.core.schemas import SegmenterResponseDTO
from semantic_msa.agents.core.telemetry import TelemetryEncoder


class SemanticSegmenterAgent(BaseAgent):
    """
    Semantic Segmenter node: synthesizes granular telemetry and micro-captions
    to discover macro-structural musical segments and write synthesized narrative captions.
    """

    def __init__(self, config: Dict[str, Any], factory: LLMFactory):
        super().__init__("segmenter", config, factory)
        self.granularity = str(config.get("granularity", "MEDIUM"))

    def get_system_prompt(self) -> str:
        base_prompt = super().get_system_prompt()
        return base_prompt.replace("{GRANULARITY}", self.granularity)

    def run(
        self,
        track_data: Dict[str, Any],
        profiler_data: Dict[str, Any],
        researcher_data: Dict[str, Any],
        token_captions: List[Dict[str, Any]],
    ) -> Tuple[str, SegmenterResponseDTO, Dict[str, Any]]:
        telemetry = TelemetryEncoder.encode_for_segmenter(track_data, token_captions)
        formatted_profiler_data = TelemetryEncoder.format_profiler_context(profiler_data)
        formatted_researcher_data = TelemetryEncoder.format_researcher_context(researcher_data)

        user_message = (
            f"\n\nRESEARCHER OUTPUT:\n"
            f"{formatted_researcher_data}\n PROFILER OUTPUT:\n"
            f"{formatted_profiler_data}\n TARGET TRACK TELEMETRY:\n"
            f"{telemetry} \n"
        )

        raw_response, parsed_dto = self.generate_and_parse(
            system_prompt=self.get_system_prompt(),
            user_message=user_message,
            schema_class=SegmenterResponseDTO,
        )

        return user_message, parsed_dto, parsed_dto.model_dump()
