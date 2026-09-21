from typing import Any, Dict, Tuple
from semantic_msa.agents.core.base import BaseAgent
from semantic_msa.agents.core.factory import LLMFactory
from semantic_msa.agents.core.schemas import ResearcherResponseDTO
from semantic_msa.agents.core.telemetry import TelemetryEncoder


class CulturalResearcherAgent(BaseAgent):
    """
    Cultural Researcher node: queries live web knowledge to ground
    the track in its historical zeitgeist and identify source mutations.
    """

    def __init__(self, config: Dict[str, Any], factory: LLMFactory):
        super().__init__("researcher", config, factory)

    def run(
        self,
        track_data: Dict[str, Any],
        profiler_data: Dict[str, Any],
    ) -> Tuple[str, ResearcherResponseDTO, Dict[str, Any]]:
        telemetry = TelemetryEncoder.encode_for_researcher(track_data)
        formatted_profiler_data = TelemetryEncoder.format_profiler_context(profiler_data)

        user_message = (
            "Here is the input payload for the target track. \n\n"
            "Before generating your JSON response, please use your web search capabilities "
            "to look up the lyrics and establish the cultural context/zeitgeist of the source material.\n\n"
            f"TARGET TRACK TELEMETRY:\n{telemetry}\n PROFILER OUTPUT:\n{formatted_profiler_data}\n"
        )

        raw_response, parsed_dto = self.generate_and_parse(
            system_prompt=self.get_system_prompt(),
            user_message=user_message,
            schema_class=ResearcherResponseDTO,
        )

        return user_message, parsed_dto, parsed_dto.model_dump()
