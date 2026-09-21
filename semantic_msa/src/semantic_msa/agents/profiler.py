from typing import Any, Dict, Tuple
from semantic_msa.agents.core.base import BaseAgent
from semantic_msa.agents.core.factory import LLMFactory
from semantic_msa.agents.core.schemas import ProfilerResponseDTO
from semantic_msa.agents.core.telemetry import TelemetryEncoder


class AcousticProfilerAgent(BaseAgent):
    """
    Acoustic Profiler node: ingests raw DSP telemetry and infers
    macro genre, energy tiers, vocal presence, and dynamic acoustic trajectory.
    """

    def __init__(self, config: Dict[str, Any], factory: LLMFactory):
        super().__init__("profiler", config, factory)

    def run(self, track_data: Dict[str, Any]) -> Tuple[str, ProfilerResponseDTO, Dict[str, Any]]:
        user_message = TelemetryEncoder.encode_for_profiler(track_data)

        raw_response = self.provider.generate(
            system_prompt=self.get_system_prompt(),
            user_message=user_message,
            temperature=self.temperature,
            grounding=self.grounding,
            response_schema=ProfilerResponseDTO,
            thinking_budget=self.thinking_budget,
        )

        parsed_dto = self.parse_response(raw_response, ProfilerResponseDTO)
        return user_message, parsed_dto, parsed_dto.model_dump()
