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

        raw_response, parsed_dto = self.generate_and_parse(
            system_prompt=self.get_system_prompt(),
            user_message=user_message,
            schema_class=ProfilerResponseDTO,
        )

        return user_message, parsed_dto, parsed_dto.model_dump()
