import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from agent import Agent
from llm_providers import LLMProvider, GeminiNativeGroundedProvider

from profiler import ProfilerResponseDTO

# TODO fix imports, this is a temporary solution to avoid circular imports, but we should refactor the codebase to avoid this in the future
current_dir = os.path.dirname(__file__)
utils_dir = os.path.abspath(os.path.join(current_dir, ".."))
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

from track import Track



logger = logging.getLogger("MSAPipeline.Researcher")

class Researcher(Agent):
    def __init__(self, api_manager: LLMProvider, explainable: bool = False):
        super().__init__(api_manager, "researcher", explainable)

    # TODO this will be useful when we will use a local model that does not support native search, but for now it is not used since Gemini can do native search grounding
    def _execute_local_web_search(self, query: str, max_results: int = 4) -> str:
        """Executes a real-time web search locally via Python text scraping."""
        pass

    def research(self,track: Track) -> str:
        """
        Coordinates the research phase, choosing dynamically whether to execute
        local web searches or rely on the LLM provider's native search grounding capabilities based on its declared features.
        """
        
        # Inspect provider capabilities to handle search logic
        if self.api_manager.supports_native_search:
            logger.info("Provider supports native search grounding. Skipping local web search execution loop.")
        else:
            logger.info("Provider lacks native search capabilities. Initiating local Python search loop...")
            # TODO 
        
        track_data = self.format_track_data(track)

        logger.info(f"Formatted track data for research: {track_data}")

        # take the profiler's output, stript it from the rationale and use it as context for this agent.

        
        user_message = track_data
        return self.api_manager.generate(
            system_prompt=self.get_system_prompt(),
            user_message=user_message,
            temperature=0.1,
        )
    
    def get_user_message(self, track: Track) -> str:
        """Formats the user message for the research phase, including relevant track information."""
        user_message = """Here is the input payload for the target track. 

Before generating your JSON response, please use your web search capabilities to look up the lyrics and establish the cultural context/zeitgeist of the source material.\n\n"""
        track_data = self.format_track_data(track)
        # TODO customize also with web context if the provider does not support native search grounding

        # read profiler output (will be automatic once we implement the orchestrator. For now we will read it from a file, but in the future it will be passed as an argument to this function) and add it to the user message as context for the research phase.

        with open(Path(track.name[:-4]) / "profiler.json", "r", encoding="utf-8") as f:
            print("Reading profiler output from:", Path(track.name[:-4]) / "profiler.json")
            profiler_output = ProfilerResponseDTO.from_dict(json.load(f))
            formatted_profiler_data = self.format_profiler_data(profiler_output)

        user_message += f"""TARGET TRACK TELEMETRY:
{track_data}\n PROFILER OUTPUT:
{formatted_profiler_data}\n"""

        return user_message

    def format_track_data(self, track: Track) -> str:
        """Extracts and formats relevant information from the track metadata for use in the research phase.
        - name 
        - author 
        - key
        - bpm 
        - duration
        - lyrics profile which contains the following information (see track.py, here we strip the metrics) :
          - lyrics 
          - reliability of the lyrics"""
        stripped_lyrics_profile = {
            "lyrics": track.LLM_lyrics_profile["text"],
            "reliability": track.LLM_lyrics_profile["reliability"]
        }

        metadata = {
            "name": track.name,
            "author": track.author,
            "key": track.key,
            "bpm": track.bpm,
            "duration": f"{track.duration_ms // 60000}:{(track.duration_ms % 60000) // 1000:02d}",
            "lyrics_profile": stripped_lyrics_profile
        }
        return json.dumps(metadata, ensure_ascii=True)
    
    def format_profiler_data(self, profiler_output: ProfilerResponseDTO):
        """Formats the profiler's output to be used as context for the researcher, so it strips the rationale and only keeps the relevant information for the research phase."""
        profiler_data = profiler_output.to_dict()
        # remove the rationale from the profiler output (and in nested structures if present) and keep only the relevant information for the research phase
        macro_profile = profiler_data.get("macro_profile", {})
        vocal_profile = profiler_data.get("vocal_profile", {})
        dynamic_trajectory = profiler_data.get("dynamic_trajectory", {})

        macro_profile.pop("rationale", None)
        vocal_profile.pop("rationale", None)
        dynamic_trajectory.pop("rationale", None)

        return {
            "macro_profile": macro_profile,
            "vocal_profile": vocal_profile,
            "dynamic_trajectory": dynamic_trajectory
        }

def resolve_path(track_name: str) -> str:
    """Resolves the full.json path for a given track name."""
    track_folder = Path(track_name).stem
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parents[2]
    return str(repo_root / "data" / "json_db" / track_folder / "full.json")


def get_wizard_of_oz(track_name: str) -> str:
    track_path = resolve_path(track_name)
    track = Track.load_from_cache(track_path)

    researcher = Researcher(api_manager=GeminiNativeGroundedProvider(), explainable=False)

    user_message = researcher.get_user_message(track)
    return researcher.get_system_prompt() + "\n\n" + user_message


if __name__ == "__main__":
    track_name = "vielleicht"
    
    print(get_wizard_of_oz(track_name))


    # track_path = resolve_path(track_name)
    # track = Track.load_from_cache(track_path)


    # researcher = Researcher(api_manager=GeminiNativeGroundedProvider(), explainable=True)

    # track_data = researcher.format_track_data(track)
    # print("Formatted track data for research:")
    # print(track_data)
    # response = []# researcher.research(track)  
    # print("Researcher response:")
    # print(json.dumps(response, indent=2))

    # output_path = Path(resolve_path(track_name)).parent / "researcher.json"
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(response, f, ensure_ascii=True, indent=2)
    # print(f"Researcher response saved to {output_path}")

    

    
