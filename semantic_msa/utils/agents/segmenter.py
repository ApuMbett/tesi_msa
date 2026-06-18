import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from agent import Agent
from llm_providers import LLMProvider, GeminiNativeGroundedProvider

from profiler import ProfilerResponseDTO, ProfilerTokenDTO

# TODO fix imports, this is a temporary solution to avoid circular imports, but we should refactor the codebase to avoid this in the future
current_dir = os.path.dirname(__file__)
utils_dir = os.path.abspath(os.path.join(current_dir, ".."))
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

from track import Track



logger = logging.getLogger("MSAPipeline.Segmenter")
GRANULARITY = "MEDIUM" #LOW, MEDIUM, HIGH


class SegmenterTokenDTO(ProfilerTokenDTO):
    def __init__(self, token_data: Dict[str, Any]):
      old_vocal_features = token_data.get("dsp_features", {}).get("vocal_features", {}).copy()  # Store the old vocal features before calling the superclass constructor
      super().__init__(token_data)
      self.caption = token_data.get("caption", "")
      # add back the vocal features to the token DTO, since they are part of the context for the segmenting phase.
      if "vocal_features" in self.dsp_features:
        self.dsp_features["vocal_features"] = old_vocal_features
        self.dsp_features.pop("vocal_edges", None)  # remove the vocal edges from the token DTO, since they are not relevant for the segmenting phase and they add unnecessary complexity to the prompt.

    def to_dict(self) -> Dict[str, Any]:
      # 1. Get the base dictionary from the Profiler
      base_dict = super().to_dict()
      
      # 2. Create a new dictionary to strictly enforce our LLM data architecture
      ordered_dict = {}
      
      # 3. Pop the index from the base and put it first
      ordered_dict["index"] = base_dict.pop("index")
      
      # 4. Insert the caption immediately after the index
      ordered_dict["caption"] = self.caption
      
      # 5. Append everything else (cue_type, lyrics, dsp_features) in their original order
      ordered_dict.update(base_dict)

      # TODO check why vocal_features: {}
      
      return ordered_dict




class Segmenter(Agent):
    def __init__(self, api_manager: LLMProvider, explainable: bool = False):
        super().__init__(api_manager, "segmenter", explainable)


    def caption(self,track: Track) -> str:
        """
        Coordinates the research phase, choosing dynamically whether to execute
        local web searches or rely on the LLM provider's native search grounding capabilities based on its declared features.
        """
        pass 


    def get_system_prompt(self) -> str:
        prompt  = super().get_system_prompt()
        return prompt.replace("{GRANULARITY}", str(GRANULARITY))
    
    def get_user_message(self, track: Track) -> str:
        user_message = """\n\n"""
        track_data = self.format_track_data(track)
        # read profiler output (will be automatic once we implement the orchestrator. For now we will read it from a file, but in the future it will be passed as an argument to this function) and add it to the user message as context for the research phase.

        with open(Path(track.name[:-4]) / "profiler.json", "r", encoding="utf-8") as f:
            profiler_output = ProfilerResponseDTO.from_dict(json.load(f))
            formatted_profiler_data = self.format_profiler_data(profiler_output)

        # read researcher output (will be automatic once we implement the orchestrator. For now we will read it from a file, but in the future it will be passed as an argument to this function) and add it to the user message as context for the captioning phase.
        with open(Path(track.name[:-4]) / "researcher.json", "r", encoding="utf-8") as f:
            researcher_output = json.load(f)
            researcher_output.pop("researcher_notes", None)  # remove the researcher's notes from the researcher output and keep it as context for the captioning phase
            print(json.dumps(researcher_output, indent=4))  # Debug statement to check the content of the researcher output
            formatted_researcher_data = researcher_output


        


        user_message += f"""RESEARCHER OUTPUT:
{formatted_researcher_data}\n PROFILER OUTPUT:
{formatted_profiler_data}\n TARGET TRACK TELEMETRY:
{track_data} \n"""

        return user_message

    def format_track_data(self, track: Track) -> str: # TODO: this is a brutal copy and paste from profiler, will be implemented in the orchestrator. 
        """Extracts and formats relevant information from the track metadata for use in the profiling phase."""
        LLM_track_metadata = track.LLM_representation()
        

        metadata = LLM_track_metadata

        def compute_global_metrics(tokens: list[Dict[str, Any]]) -> Dict[str, float]:
            
            collect_values = lambda key: [float(token["dsp_features"][key]) for token in tokens[1:-1]] # exclude the head and tail tokens
            safe_max = lambda values: round(max(values), 2)
            safe_avg = lambda values: round(sum(values) / len(values), 2)

            msf_values = collect_values("MSF")
            btr_values = collect_values("BTR")
            roe_values = collect_values("ROE")
            her_values = collect_values("HER")
            sf_values = collect_values("SF")
            ocn_values = collect_values("OCN")

            return {
                "peak_msf": safe_max(msf_values),
                "peak_btr": safe_max(btr_values),
                "peak_roe": safe_max(roe_values),
                "avg_her": safe_avg(her_values),
                "avg_sf": safe_avg(sf_values),
                "avg_btr": safe_avg(btr_values),
                "avg_ocn": safe_avg(ocn_values),
            }

        # we format the duration in minutes and seconds to make it easier for the LLM to understand and reason about the structure of the track without getting overwhelmed by too many numerical details
        duration_ms = metadata.get("duration_ms", 0)
        formatted_duration = f"{duration_ms // 60000}:{(duration_ms % 60000) // 1000:02d}"

        

        # for each token we strip the cue point timing information and replace it with the cue type (START, END, BOTH, NONE) to make it easier for the LLM to understand and reason about the structure of the track without getting overwhelmed by too many numerical details
        for i, token in enumerate(metadata["tokens"]):
            # TODO refactor, this is a brutal copy and paste from captioner, will be implemented in the orchestrator.
            # add caption field to each token after reading the captioner output 
            with open(Path(track.name[:-4]) / "captioner.json", "r", encoding="utf-8") as f:
                captioner_output = json.load(f)
                token_captions = captioner_output.get("token_captions", [])
                metadata["tokens"][i]["caption"] = token_captions[i].get("caption", "") 

            
            token_dto = SegmenterTokenDTO(token)
            print(f"###### TOKEN {i} ######")
            # Update the actual item in the list with your cleaned DTO dictionary
            metadata["tokens"][i] = token_dto.to_dict()

            print("CLEANED TOKEN DATA:")
            print(json.dumps(metadata["tokens"][i], indent=2))  # Debug statement to





            
        ordered_metadata = {
            "name": metadata.get("name"),  
            "author": metadata.get("author"),
            # in this case we want to keep the original author and track name, since they are part of the context for the captioning phase.
            "duration": formatted_duration,
            "default_token_beats": metadata.get("token_size_beats", 16), 
            "key": metadata.get("key"),
            "global_metrics": compute_global_metrics(LLM_track_metadata.get("tokens", [])), 
            "tokens": metadata.get("tokens", [])
        }    

        # TODO add global vocal features.
        # TODO edit the prompt to support global features "Use the global_vocal_intensity and global_vocal_density to determine the overall presence_level, but rely on the individual token math to map where those vocals physically start and stop in the dynamic_trajectory."
        return json.dumps(ordered_metadata, ensure_ascii=True)
    
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

    segmenter = Segmenter(api_manager=GeminiNativeGroundedProvider(), explainable=False)

    user_message = segmenter.get_user_message(track)
    return segmenter.get_system_prompt() + "\n\n" + user_message


if __name__ == "__main__":
    track_name = "shotmedown"
    
    print(get_wizard_of_oz(track_name))
    

    
