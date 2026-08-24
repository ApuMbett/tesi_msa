import json
from typing import Any, Dict, List


  # temporarily read from jsons, will read from the real outputs in the future
from pathlib import Path

BASE_DIR = Path(__file__).parent  # .../semantic_msa/utils/agents/
PROJECT_ROOT = BASE_DIR.parent.parent.parent  # .../tesi/

class TrackDTO:
    def __init__(self, track_data: Dict[str, Any], unit = "seconds"):
        self.track_identity = TrackIdentityDTO(track_data)
        self.unit = unit
        self.token_size_beats = track_data.get("token_size_beats", 0)
        self.lyrics_profile = track_data.get("lyrics_profile", {})
        self.caption_length = track_data.get("captioner", 0).get("CAPTION_LENGTH", 0)
        self.segmenter_granularity = track_data.get("segmenter", {}).get("GRANULARITY", "MEDIUM")

        # build the tokens with the captions 
        self.tokens = []
        captions = track_data.get("captioner",{}).get("token_captions", [])
        for i, token in enumerate(track_data.get("tokens", [])):
            # map the caption 
            caption = captions[i].copy()
            caption.pop("token_index", None)  # remove the token index from the caption, it's redundant
            token["captioning_data"] = caption

            token_dto = TokenDTO(token, unit=self.unit)

            self.tokens.append(token_dto)
        

        # build the segments and put tokens inside them

        self.segments = [SegmentDTO(segment, self.tokens) for segment in track_data.get("segmenter", []).get("segments", [])]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_identity": self.track_identity.to_dict(),
            "token_size_beats": self.token_size_beats,
            "lyrics_profile": self.lyrics_profile,
            "caption_length": self.caption_length,
            "segmenter_granularity": self.segmenter_granularity,
            "segments": [segment.to_dict() for segment in self.segments],
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)
    

class TrackIdentityDTO:
    def __init__(self, track_data: Dict[str, Any]):
        self.name = track_data.get("name", "")
        self.artist = track_data.get("artist", "")
        self.key = track_data.get("key", "")
        self.bpm = track_data.get("bpm", 0)
        duration_ms = track_data.get("duration_ms", 0.0)
        self.duration = f"{duration_ms // 60000}:{(duration_ms % 60000) // 1000:02d}"
        self.researcher = track_data.get("researcher", "")
        self.profiler = track_data.get("profiler", "")
        # self.album = identity_data.get("album", "")
        # self.release_date = identity_data.get("release_date", "")
        # self.genre = identity_data.get("genre", "")

    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "artist": self.artist,
            "key": self.key,
            "bpm": self.bpm,
            "duration": self.duration,
            "cultural_context": self.researcher,
            "acoustic_profile": self.profiler,
            # "album": self.album,
            # "release_date": self.release_date,
            # "genre": self.genre,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)



class TokenDTO:
    def __init__(self, token_data: Dict[str, Any], unit = "seconds"):
        self.index = token_data.get("index", 0)
        self.start_time = self._to_unit(token_data.get("start_time_ms", 0.0), unit)
        self.end_time = self._to_unit(token_data.get("end_time_ms", 0.0),unit)


        self.start_cue = token_data.get("start_cue", {})
        self.end_cue = token_data.get("end_cue", {})

        # apply unit conversion to cue points if they exist
        if self.start_cue:
            self.start_cue["time"] = self._to_unit(self.start_cue.get("time_ms", 0.0), unit)
        if self.end_cue:
            self.end_cue["time"] = self._to_unit(self.end_cue.get("time_ms", 0.0), unit)

        self.lyrics = token_data.get("lyrics", "")
        # apply unit conversion to lyrics timestamps if they exist
        if self.lyrics:
          self.lyrics["start_time"] = self._to_unit(self.lyrics.get("start_time_ms", 0.0), unit)
          self.lyrics["end_time"] = self._to_unit(self.lyrics.get("end_time_ms", 0.0), unit)


        self.dsp_features = token_data.get("dsp_features", {})

        self.caption = token_data.get("captioning_data", "")


    def _to_unit(self, value_ms: float, unit: str = "seconds") -> float:
        """Convert the value to the specified unit."""
        if unit == "milliseconds":
            return value_ms
        elif unit == "seconds":
            return value_ms / 1000
        elif unit == "minutes":
            return f"{int(value_ms // 60000)}:{int((value_ms % 60000) // 1000):02d}"
        else:
            raise ValueError(f"Unsupported unit: {unit}")
        
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "index": self.index,
            "captioning_data": self.caption,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        # sparse representation of cue points, only include them if they exist
        if self.start_cue:
            d["start_cue"] = self.start_cue
        if self.end_cue:
            d["end_cue"] = self.end_cue

        if self.lyrics:
            d["lyrics"] = self.lyrics

        
        d["dsp_features"] = self.dsp_features
        

        return d
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

class SegmentDTO:
    def __init__(self, segment_data: Dict[str, Any], tokens: list[TokenDTO]):
        self.index = segment_data.get("segment_index", 0)
        self.name = segment_data.get("segment_name", "")
        self.macro_caption = segment_data.get("macro_caption", "")
        self.rationale = segment_data.get("rationale", "")
        token_range = segment_data.get("token_range", [0, 0])
        self.tokens = [token for token in tokens if token_range[0] <= token.index <= token_range[1]]

        # for ease of access, we can also include the start and end time of the segment based on the tokens it contains
        if self.tokens:
            self.start_time = self.tokens[0].start_time
            self.end_time = self.tokens[-1].end_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "macro_caption": self.macro_caption,
            "rationale": self.rationale,
            "tokens": [token.to_dict() for token in self.tokens],
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)



class Orchestrator:
    

    def build_full_json(self, track_name):


      with open(PROJECT_ROOT / "data" / "json_db" / track_name / "LLM.json", "r") as f:
          full_data = json.load(f)

      # the rest live in {track_name}/ next to orchestrator.py
      track_dir = BASE_DIR / track_name

      with open(track_dir / "researcher.json", "r") as f:
          full_data["researcher"] = json.load(f)

      with open(track_dir / "profiler.json", "r") as f:
          full_data["profiler"] = json.load(f)

      with open(track_dir / "captioner.json", "r") as f:
          full_data["captioner"] = json.load(f)

      with open(track_dir / "segmenter.json", "r") as f:
          full_data["segmenter"] = json.load(f)

        
      track = TrackDTO(full_data, "minutes")

      return track.to_json()

      
      



if __name__ == "__main__":
    orchestrator = Orchestrator()
    track_name = "satisfaction"  
    track_json = orchestrator.build_full_json(track_name) 

    # dump json to <track_name>/final.json

    # create file if it doesn't exist, otherwise overwrite it


    with open(PROJECT_ROOT / "data" / "json_db" / track_name / "final.json", "w") as f:
        f.write(track_json)
     


        
        