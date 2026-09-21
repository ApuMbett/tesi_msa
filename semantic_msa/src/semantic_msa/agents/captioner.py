from typing import Any, Dict, Tuple
from semantic_msa.agents.core.base import BaseAgent
from semantic_msa.agents.core.factory import LLMFactory
from semantic_msa.agents.core.schemas import CaptionerResponseDTO
from semantic_msa.agents.core.telemetry import TelemetryEncoder


class TokenCaptionerAgent(BaseAgent):
    """
    Token Captioner node: generates dense, context-aware acoustic captions
    for every individual 16-beat token in the track in a single batched inference call.
    """

    def __init__(self, config: Dict[str, Any], factory: LLMFactory):
        super().__init__("captioner", config, factory)
        self.caption_length = int(config.get("caption_length", 12))

    def get_system_prompt(self) -> str:
        base_prompt = super().get_system_prompt()
        return base_prompt.replace("{CAPTION_LENGTH}", str(self.caption_length))

    def run(
        self,
        track_data: Dict[str, Any],
        profiler_data: Dict[str, Any],
        researcher_data: Dict[str, Any],
    ) -> Tuple[str, CaptionerResponseDTO, Dict[str, Any]]:
        telemetry = TelemetryEncoder.encode_for_captioner(track_data)
        formatted_profiler_data = TelemetryEncoder.format_profiler_context(profiler_data)
        formatted_researcher_data = TelemetryEncoder.format_researcher_context(researcher_data)

        user_message = (
            f"\n\nRESEARCHER OUTPUT:\n"
            f"{formatted_researcher_data}\n PROFILER OUTPUT:\n"
            f"{formatted_profiler_data}\n TARGET TRACK TELEMETRY:\n"
            f"{telemetry} \n"
        )

        raw_response = self.provider.generate(
            system_prompt=self.get_system_prompt(),
            user_message=user_message,
            temperature=self.temperature,
            grounding=self.grounding,
            response_schema=CaptionerResponseDTO,
            thinking_budget=self.thinking_budget,
        )

        parsed_dto = self.parse_response(raw_response, CaptionerResponseDTO)
        return user_message, parsed_dto, parsed_dto.model_dump()
