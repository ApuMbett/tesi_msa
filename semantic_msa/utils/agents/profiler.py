import logging
import os
import sys
import json
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional
from agent import Agent
from llm_providers import LLMProvider, GeminiNativeGroundedProvider

# TODO fix imports, this is a temporary solution to avoid circular imports, but we should refactor the codebase to avoid this in the future
current_dir = os.path.dirname(__file__)
utils_dir = os.path.abspath(os.path.join(current_dir, ".."))
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

from track import Track



logger = logging.getLogger("MSAPipeline.Profiler")

class ProfilerTokenDTO:
    def __init__(self, token: Dict[str, Any]):
        
        self.index = token.get("index")
        self.cue_type = self.format_cue_type(
            token.get("start_cue", False),
            token.get("end_cue", False),
        )
        self.dsp_features = token.get("dsp_features", {}).copy()  # Use copy to avoid mutating the original token's DSP features

        if "vocal_features" in self.dsp_features:    
          # TODO refactor: the ideal way to do this would be to have a separate DSPFeaturesDTO that would handle the formatting of the DSP features.
          formatted_vocal_edges = self.format_vocal_edges(self.dsp_features["vocal_features"])
          self.dsp_features["vocal_edges"] = formatted_vocal_edges
          self.dsp_features["vocal_features"].pop("start_edge", None)
          self.dsp_features["vocal_features"].pop("end_edge", None)
        self.lyrics = token.get("lyrics", "")

    def format_cue_type(self, start_cue, end_cue) -> str:
        if start_cue and end_cue:
            return CueType.BOTH.value
        elif start_cue:
            return CueType.START.value
        elif end_cue:
            return CueType.END.value
        else:
            return CueType.NONE.value

    def format_vocal_edges(self, vocal_features) -> str:
        
        start_edge = vocal_features.get("start_edge", False)
        end_edge = vocal_features.get("end_edge", False)
        

        # TODO recfactor this into has_vocals function in dsp_manager.py  
        MIN_CONFIDENCE = 0.50
        MIN_DENSITY = 0.15      # Must occupy at least 15% of the edge
        MIN_INTENSITY = 0.0     # if intensity is > 0 if there are vocals, see dsp_manager.py 

        def is_active(vocals: dict) -> bool:
            return (
                vocals["confidence"] > MIN_CONFIDENCE and 
                vocals["density"] > MIN_DENSITY and 
                vocals["intensity"] > MIN_INTENSITY
            )
        # Note: i could have also designed an aggregated vocal_activity_score that combines the confidence, density and intensity into a single score to determine if the edge is active or not, but for now this simple thresholding approach should work well enough and is easier to interpret and explain.
        # END TODO refactor
        if is_active(start_edge) and is_active(end_edge):
            return VocalEdges.BOTH.value
        elif is_active(start_edge):
            return VocalEdges.START.value
        elif is_active(end_edge):
            return VocalEdges.END.value
        else:
            return VocalEdges.NONE.value
    
    def to_dict(self) -> Dict[str, Any]:
      dsp_features = {}
      if "BPM" in self.dsp_features:
          dsp_features["BPM"] = self.dsp_features["BPM"]

      if "vocal_features" in self.dsp_features:
          dsp_features["vocal_features"] = self.dsp_features["vocal_features"]

      vocal_edges = self.dsp_features.get("vocal_edges")
      if vocal_edges and vocal_edges != VocalEdges.NONE.value:
          dsp_features["vocal_edges"] = vocal_edges

      micro_math_keys = [
          "ROE",
          "BTR",
          "MTR",
          "TTR",
          "SC",
          "MSF",
          "SF",
          "OCN",
          "EV",
          "HER",
      ]
      for key in micro_math_keys:
          if key in self.dsp_features:
              dsp_features[key] = self.dsp_features[key]

      d = {
          "index": self.index,
      }

      if self.cue_type != CueType.NONE.value:
          d["cue_type"] = self.cue_type

      if self.lyrics:
          d["lyrics"] = self.lyrics

      d["dsp_features"] = dsp_features
      return d
        
    
    
class CueType(str, Enum):
    START = "START"  # Has a start cue only.
    END = "END"  # Has an end cue only.
    BOTH = "BOTH"  # Has both (e.g., a 16-beat loop drop).
    NONE = "NONE"  # No cues present.


class VocalEdges(str, Enum):
    NONE = "NONE"  # Vocals are fully contained inside the token, or there are no vocals.
    START = "START"  # Vocals are active in the first window.
    END = "END"  # Vocals carry over into the last window.
    BOTH = "BOTH"  # Vocals span across the entire token.
    # Note: for window size check dsp_manager.py 

class Profiler(Agent):
    def __init__(self, api_manager: LLMProvider, explainable: bool = False):
        super().__init__(api_manager, "profiler", explainable)

    def profile(self,track: Track) -> str:
        """
        Profiles the track by extracting relevant metadata and generating a structured representation of the track's characteristics, including its lyrics profile, using the LLM provider's capabilities
        """
        
        track_data = self.format_track_data(track)

        logger.info(f"Formatted track data for research: {track_data}")
        
        user_message = track_data
        return self.api_manager.generate(
            system_prompt=self.get_system_prompt(),
            user_message=user_message,
            temperature=0.1,
        )
    
    def get_user_message(self, track: Track) -> str:
        """Formats the user message for the research phase, including relevant track information."""
        track_data = self.format_track_data(track)
        return track_data

    def format_track_data(self, track: Track) -> str:
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
            token_dto = ProfilerTokenDTO(token)
            # Update the actual item in the list with your cleaned DTO dictionary
            metadata["tokens"][i] = token_dto.to_dict()

            
        ordered_metadata = {
            "duration": formatted_duration,
            "default_token_beats": metadata.get("token_size_beats", 16), 
            "key": metadata.get("key"),
            "global_metrics": compute_global_metrics(LLM_track_metadata.get("tokens", [])),
            "tokens": metadata.get("tokens", [])
        }    
        return json.dumps(ordered_metadata, ensure_ascii=True)

def resolve_path(track_name: str) -> str:
    """Resolves the full.json path for a given track name."""
    track_folder = Path(track_name).stem
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parents[2]
    return str(repo_root / "data" / "json_db" / track_folder / "full.json")


def get_wizard_of_oz(track_name: str) -> str:
    track_path = resolve_path(track_name)
    track = Track.load_from_cache(track_path)

    profiler = Profiler(api_manager=GeminiNativeGroundedProvider(), explainable=True)
    track_data = profiler.format_track_data(track)

    user_message = f"""TARGET TRACK TELEMETRY:
{track_data}"""

    return profiler.get_system_prompt() + "\n\n" + user_message

if __name__ == "__main__":
    track_name = "intheend"
    output_text = get_wizard_of_oz(track_name)
    output_dir = Path(__file__).resolve().parent / Path(track_name).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "oz.md"
    output_path.write_text(output_text, encoding="utf-8")


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

    

    
