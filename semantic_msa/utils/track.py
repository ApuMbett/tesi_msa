from dsp_manager import DSPManager
from cue_point import RawCuePoint, SnappedCuePoint
import json
from pathlib import Path
import subprocess
import sys
HIGH_MEDIAN_SCORE = 0.8
HIGH_MIN_SCORE = 0.6
LOW_MEDIAN_SCORE = 0.5
LOW_MIN_SCORE = 0.2
CUE_DETR_SENSITIVITY = 0.85
class Token: 
  def __init__(self, index, start_time_ms, end_time_ms, is_mixable, dsp_features, lyrics, start_cue = None, end_cue = None):
    self.index = index
    self.start_time_ms = start_time_ms
    self.end_time_ms = end_time_ms
    self.is_mixable = is_mixable
    self.start_cue = start_cue
    self.end_cue = end_cue

    self.tags = None
    self.caption = None # TODO this will be generated later by the captioning model, it's not important for now
    self.dsp_features = dsp_features # TODO this will hold the dsp features for the token, such as bpm, key, energy, etc. we can compute these later using the beatnet wrapper and other tools. it's not important for now
    self.lyrics = lyrics

  def get_audio_segment(self):
    # TODO this will return the audio segment corresponding to the token, pydub to do this. it's not important for now
    pass

  # TODO  
  def _get_cue_data_LLM_representation(self):
     # just a placeholder. move it into cue_point.py
     pass
  def _get_lyrics_LLM_representation(self):
    if self.lyrics:
      words = self.lyrics["words"]
      # we preprocess lyrics for LLM token economy.
      text_lyrics = " ".join([word["word"] for word in words])
      # if a word spans across token boundaries, we add a special token to indicate that the word bleeds into the next token.
      if words[-1]["end"] * 1000 > self.end_time_ms:
        bleed_ms = int(words[-1]["end"] * 1000 - self.end_time_ms)
        text_lyrics += f" <BLEED {bleed_ms}>"

      # we save the start and the end time of the lyrics. useful because there could be instrumental padding useful for the mix
      start = int(words[0]["start"] * 1000)
      end = int(words[-1]["end"] * 1000) # <-- the LLM can see that the word spans across the token boundary also by looking at the end time.

      # we add confidence scores for the lyrics
      if self.lyrics["median_score"] < LOW_MEDIAN_SCORE or self.lyrics["min_score"] < LOW_MIN_SCORE:
        reliability_tag = "LOW (Hallucination Risk)"
      elif self.lyrics["median_score"] >= HIGH_MEDIAN_SCORE and self.lyrics["min_score"] >= HIGH_MIN_SCORE:
        reliability_tag = "HIGH"
      else:
        reliability_tag = "MEDIUM"

      return {
        "text": text_lyrics,
        "start_time_ms": start,
        "end_time_ms": end,
        "reliability": reliability_tag
      }
      



  def json(self):
    return {
      "index": self.index,
      "start_time_ms": self.start_time_ms,
      "end_time_ms": self.end_time_ms,
      "is_mixable": self.is_mixable,
      "start_cue": self.start_cue.json() if self.start_cue else None,
      "end_cue": self.end_cue.json() if self.end_cue else None,
      "caption": self.caption,
      "dsp_features": self.dsp_features,
      "lyrics": self.lyrics,
      "tags": self.tags
    }
  # TODO refactor classes as dataclasses
  @classmethod
  def from_dict(cls, data):
    return cls(
        index=data["index"],
        start_time_ms=data["start_time_ms"],
        end_time_ms=data["end_time_ms"],
        is_mixable=data["is_mixable"],
        dsp_features=data.get("dsp_features", {}),
        lyrics=data.get("lyrics", None),
        start_cue=RawCuePoint.from_dict(data["start_cue"]) if data.get("start_cue") else None,
        end_cue=RawCuePoint.from_dict(data["end_cue"]) if data.get("end_cue") else None
    )
  def LLM_representation(self):
    return {
      "index": self.index,
      "start_time_ms": self.start_time_ms,
      "end_time_ms": self.end_time_ms,
      "is_mixable": self.is_mixable,
      "start_cue": self.start_cue.json() if self.start_cue else None,
      "end_cue": self.end_cue.json() if self.end_cue else None,
      "caption": self.caption,
      "dsp_features": self._get_dsp_features_LLM_representation(),
      "lyrics": self._get_lyrics_LLM_representation() if self.lyrics else None,
      "tags": self.tags
    }
  
  def _get_dsp_features_LLM_representation(self):
    # deep copy to avoid mutating the original features
    dsp_features = self.dsp_features.copy()

    # Round top-level numeric features for readability.
    for key, value in dsp_features.items():
      if isinstance(value, float):
        dsp_features[key] = round(value, 2)

    def _vocal_features_LLM_representation(features: dict) -> dict:
      representation = features.copy()

      # Drop VED because it's not normalized and not very interpretable.
      representation.pop("vocal_energy_dominance", None)

      rename_map = {
        "vocal_intensity": "intensity",
        "vocal_confidence": "confidence",
        "vocal_density": "density",
      }
      for old_name, new_name in rename_map.items():
        if old_name in representation:
          representation[new_name] = representation.pop(old_name)

      for vocal_key in ["intensity", "confidence", "density"]:
        if isinstance(representation.get(vocal_key), float):
          representation[vocal_key] = round(representation[vocal_key], 2)

      return representation

    vocal_features = dsp_features.get("vocal_features")
    if isinstance(vocal_features, dict):
      vocal_features = vocal_features.copy()
      vocal_features["start_edge"] = _vocal_features_LLM_representation(
        vocal_features.pop("start_edge_features", {})
      )
      vocal_features["end_edge"] = _vocal_features_LLM_representation(
        vocal_features.pop("end_edge_features", {})
      )
      vocal_features = _vocal_features_LLM_representation(vocal_features)
      dsp_features["vocal_features"] = vocal_features

    return dsp_features

class Track:
  def __init__(self, path, name, author, beats_per_token = 16, structural_penalty_weight = 0.5):
    self.path = path
    self.name = name
    self.author = author

    self.dsp_manager = DSPManager(audio_path = path, beats_per_token = beats_per_token)
    self.raw_cue_points = self._compute_raw_cue_points()


    # when we compute the token boundaries we find the phase offset that maximizes the score. the cue points are snapped to downbeats in a way that the mean score is maximized.
    self.best_beat_phase_offset = None
    self.mean_cue_score = None
    self.token_boundaries = []
    self.snapped_cues = []

    self._compute_snapped_cue_points(structural_penalty_weight = structural_penalty_weight)
    self.duration_ms = self.dsp_manager.beats[-1][0]

    self.lyrics = self.dsp_manager.get_lyrics()
    
    self.vocal_density_threshold = self.dsp_manager.vocal_threshold
    self.vocal_confidence_steepness = self.dsp_manager.vocal_confidence_steepness
    self.tokens = self._build_token_map()

    self.key = self.dsp_manager.compute_camelot_key(0, self.duration_ms)


    # global vocal features 
    edge_keys = (
      "vocal_energy_dominance",
      "vocal_intensity",
      "vocal_confidence",
      "vocal_density",
    )
    global_vocal = self.dsp_manager._compute_DSP_vocal_features(0, self.duration_ms)
    self.global_vocal_features = {k: round(float(v), 4) for k, v in zip(edge_keys, global_vocal)} #TODO refactor this duplicated code. 
    self.global_vocal_features["vocal_density_threshold"] =  round(self.vocal_density_threshold, 4)
    self.global_vocal_features["vocal_confidence_steepness"] = round(self.vocal_confidence_steepness,4)
    from dsp_manager import VOCAL_START_END_BAR_WINDOW
    self.global_vocal_features["vocal_edge_window_bars"] = VOCAL_START_END_BAR_WINDOW

    tokens_bpm = [token.dsp_features.get("BPM") for token in self.tokens if token.is_mixable and token.dsp_features.get("BPM") is not None]
    self.bpm = self.dsp_manager.compute_global_bpm(cached_token_bpms=tokens_bpm)

    # generate token tags with TinyMU
    _repo_root = Path(__file__).resolve().parents[2]  # .../utils -> .../semantic_msa -> root
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from tagging_engine.tagger import Tagger

    with Tagger(python311="python3.11", ) as tagger:   # model loads once here
      for token in self.tokens:
        token_audio_path = self.generate_token_audio(token.index)
        print(f"Generated audio slice for token {token.index} at {token_audio_path}")

        print(f"Generating tags for token {token.index} using TinyMU...")
        tags = tagger.tag(token_audio_path)        # fast from here on
        print(tags)  # ["electronic", "upbeat", "synthesizer", "120bpm"]
        token.tags = tags
      

  def _compute_raw_cue_points(self) -> list[RawCuePoint]:
    base_dir = Path(__file__).resolve().parent  # semantic_msa/utils
    semantic_msa_dir = base_dir.parent
    repo_root = semantic_msa_dir.parent

    cue_detr_dir = semantic_msa_dir / "externals" / "cue-detr"
    cue_detr_script = cue_detr_dir / "cue_points_single_track.py"
    track_folder = Path(self.name).stem
    cue_points_path = repo_root / "data" / "json_db" / track_folder / "_cue_points.json"

    def _load_cue_payload(path: Path):
      with open(path, "r") as f:
        data = json.load(f)
      if isinstance(data, dict):
        return data.get("sensitivity"), data.get("cue_points", [])
      return None, data

    cue_points = []
    saved_sensitivity = None

    if cue_points_path.exists():
        print(f"Cue points already exist for {self.name}, loading from {cue_points_path}")
        saved_sensitivity, cue_points = _load_cue_payload(cue_points_path)
        print("Cue points loaded successfully.")

    if saved_sensitivity != CUE_DETR_SENSITIVITY:
        if saved_sensitivity is not None:
            print(
                f"Cue points for {self.name} were computed with a different sensitivity "
                f"({saved_sensitivity}) than the current one ({CUE_DETR_SENSITIVITY})."
            )
            print("Recomputing cue points with the current sensitivity...")
        
        cue_points_path.parent.mkdir(parents=True, exist_ok=True)
        cue_detr_python = cue_detr_dir / ".venv" / "bin" / "python"
        subprocess.run(
            [
                str(cue_detr_python),
                str(cue_detr_script),
                "--track-name",
                self.name,
                "--sensitivity",
                str(CUE_DETR_SENSITIVITY),
            ],
            check=True,
        )
        _, cue_points = _load_cue_payload(cue_points_path)
        print("Cue points computed and saved successfully.")

    # note: the cue points in the json file are in seconds, we need to convert them to milliseconds for the RawCuePoint class
    return [RawCuePoint(time_ms = int(cue["time"] * 1000), score = cue["score"]) for cue in cue_points]
    

  def _compute_snapped_cue_points(self, structural_penalty_weight = 0.5) -> list[SnappedCuePoint]:
    for i in range(self.dsp_manager.beats_per_token//4):
        snapped_cues = [SnappedCuePoint(raw_cue, self.dsp_manager, downbeat_offset_to_skip = i, structural_penalty_weight = structural_penalty_weight) for raw_cue in self.raw_cue_points]
        #! ^ this will be refactored later, snappedcuepoint should not take the whole beat grid as an argument now that the track class exists 

        # maximize the mean score across all snapped cues
        mean_cue_score = sum(cue.score for cue in snapped_cues) / len(snapped_cues)
        if self.mean_cue_score is None or mean_cue_score > self.mean_cue_score:
            self.mean_cue_score = mean_cue_score
            self.best_beat_phase_offset = i
            self.snapped_cues = snapped_cues
            self.token_boundaries = self.dsp_manager.get_semantic_audio_token_boundaries(downbeat_offset_to_skip = i)
            #!^ problem: token boundaries now include tail and intro pad, this was necessary for completeness. now we can move this to the track class, this is necessary because (see blablabla) we have the micro intro pad that has 8000 bpm and so the qer is very high. even though it's few ms 
        
        print(f"\n\n\n\n######## Phase offset {i}: mean score={mean_cue_score} ########") 
        print(F"token boundaries={self.token_boundaries}") 
        print("Snapped Cue Points for this phase offset:")
        for snapped_cue in snapped_cues:
            print(snapped_cue.json())

  def _build_token_map(self) -> list[Token]:
    boundaries = [b for b in self.token_boundaries]  # Make a copy to avoid modifying the original list of token boundaries

    # Add intro and tail padding to ensure we map the full track, theese segments though won't be mixable, they will just be used for completeness and to make sure we don't miss any cue points that are close to the start or the end of the track.
    # 1. Intro Pad: Insert 0 only if the first downbeat isn't already at 0
    if boundaries and boundaries[0] > 0:
        boundaries.insert(0, 0)
    
    # 2. Tail Pad: Append the last beat only if it isn't already captured
    if boundaries and boundaries[-1] != self.duration_ms:
        boundaries.append(self.duration_ms)

    tokens = []

    # Iterate through the boundaries to create tokens
    for i in range(len(boundaries) - 1):
        start_time_ms = boundaries[i]
        end_time_ms = boundaries[i + 1]

        # Determine if the token is mixable based on its position
        is_mixable = (i > 0 and i < len(boundaries) - 2)  # Only tokens that are not intro or tail pads are mixable

        # Determine if the token contains any snapped cue points and if so, check if they are at the start or the end of the token. 
        start_cue, end_cue = self._determine_token_cues(start_time_ms, end_time_ms)

        # Map lyrics to the token based on the token boundaries. we assign to the token all the words that are in the token boundaries.
        token_lyrics = self._map_lyrics_to_token(start_time_ms, end_time_ms)

        # calculate median and minimum word score for the token lyrics.
        if token_lyrics:
          word_scores = [word["score"] for word in token_lyrics]
          lyrics_median_score = sorted(word_scores)[len(word_scores) // 2]  # median
          lyrics_min_score = min(word_scores)
          token_lyrics = {
            "words": token_lyrics,
            "median_score": lyrics_median_score,
            "min_score": lyrics_min_score
          }
          # low min score but high median means that there are some hallucinated words for example

        # TODO DSP features
        dsp_features = {}
        if is_mixable:
          # compute the bpm only if it's not a pad 
          dsp_features.update(self.dsp_manager.compute_low_level_dsp_features(start_time_ms, end_time_ms)) 

        # TODO captioning 
        # TODO lyrics and vocal features
        token = Token(i, start_time_ms, end_time_ms, is_mixable, dsp_features, token_lyrics, start_cue = start_cue, end_cue = end_cue)
        tokens.append(token)

    # fix pad bpm 
    # if it's a padding token we just inherit the bpm from the closest real token. 
    # 1. Fix Intro Pad (It borrows from the token immediately after it)
    if tokens and not tokens[0].is_mixable:
      tokens[0].dsp_features["BPM"] = tokens[1].dsp_features["BPM"]
            
    # 2. Fix Outro Pad (It borrows from the token immediately before it)
    if len(tokens) > 1 and not tokens[-1].is_mixable:
        tokens[-1].dsp_features["BPM"] = tokens[-2].dsp_features["BPM"]

    



    return tokens

  def _determine_token_cues(self, start_time_ms, end_time_ms):
    # Check if any snapped cue points fall within the token boundaries
    token_cues = [cue for cue in self.snapped_cues if start_time_ms <= cue.time <= end_time_ms]
    
    # Determine if there are cues at the start or end of the token
    start_cue = None
    end_cue = None
    for cue in token_cues:
        if cue.time == start_time_ms:
            start_cue = cue
        elif cue.time == end_time_ms:
            end_cue = cue
    
    return start_cue, end_cue
  
  def _map_lyrics_to_token(self, start_time_ms, end_time_ms):
    # this method maps the lyrics to the tokens. if a word STARTS in the token boundaries, we assign it to the token.
    # we do this because we want to keep also words that span across token boundaries. 
    words = self.lyrics["word_segments"]
    token_lyrics = [word for word in words if start_time_ms <= word["start"] * 1000 < end_time_ms]  # convert to ms and check if it falls within the token boundaries
    
    return token_lyrics


  ##### SERIALIZATION AND LLM REPRESENTATION METHODS #####
  def generate_token_audio(self, token_index):
    if token_index < 0 or token_index >= len(self.tokens):
        raise ValueError("Invalid token index")

    token = self.tokens[token_index]
    repo_root = Path(__file__).resolve().parent.parent.parent
    track_folder = Path(self.name).stem
    tokens_dir = repo_root / "data" / "raw_audio" / track_folder / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    output_path = tokens_dir / f"{token_index}.wav"
    self.dsp_manager.generate_audio_slice(token.start_time_ms, token.end_time_ms, str(output_path))
    return str(output_path)

  def json(self):
    return {
      "path": self.path,
      "name": self.name,
      "author": self.author,
      "key": self.key,
      "bpm": self.bpm,
      "duration_ms": self.duration_ms,
      "token_size_beats": self.dsp_manager.beats_per_token,
      "best_beat_phase_offset": self.best_beat_phase_offset,
      "mean_cue_score": self.mean_cue_score,
      "vocal_features": self.global_vocal_features,
      "lyrics_profile": self.LLM_lyrics_profile,
      "token_boundaries": self.token_boundaries,
      "snapped_cues": [cue.json() for cue in self.snapped_cues],
      "tokens": [token.json() for token in self.tokens]
    }
  # TODO refactor classes as dataclasses
  # @classmethod
  # def from_dict(cls, data):
  #   track = cls(
  #       path=data["path"],
  #       name=data["name"],
  #       author=data["author"],
  #       beats_per_token=16,  # You can adjust this if needed
  #       structural_penalty_weight=0.5  # You can adjust this if needed
  #   )
  #   track.key = data.get("key", "Unknown Key")
  #   track.duration_ms = data.get("duration_ms", 0)
  #   track.best_beat_phase_offset = data.get("best_beat_phase_offset", None)
  #   track.mean_cue_score = data.get("mean_cue_score", None)
  #   track.global_vocal_features = data.get("vocal_features", {})
  #   track.token_boundaries = data.get("token_boundaries", [])
  #   track.snapped_cues = [SnappedCuePoint.from_dict(cue) for cue in data.get("snapped_cues", [])]
  #   track.tokens = [Token.from_dict(token) for token in data.get("tokens", [])]
    
  #   return track

  # TODO quick and dirty method, i will have to refactor this later by splitting the constructors and the pipeline execution. this is just to have something working for the prototype
  @classmethod
  def load_from_cache(cls, json_path: str):
    """
    Alternative Constructor: Instantiates the Track directly from full.json
    by bypassing the heavy __init__ pipeline loop using __new__, while
    preserving the exact instance attribute schema layout recursively.
    """
    # 1. Allocate space for a clean Track instance shell without running __init__
    obj = cls.__new__(cls)
    
    # 2. Parse the cache file into a plain dict
    with open(json_path, "r", encoding="utf-8") as f:
      cached_tree = json.load(f)
    
    # 3. Re-assign the properties straight to the object to preserve interface symmetry
    obj.path = cached_tree.get("path")
    obj.name = cached_tree.get("name")
    obj.author = cached_tree.get("author")
    obj.key = cached_tree.get("key")
    obj.duration_ms = cached_tree.get("duration_ms")
    obj.best_beat_phase_offset = cached_tree.get("best_beat_phase_offset")
    obj.mean_cue_score = cached_tree.get("mean_cue_score")
    obj.token_boundaries = cached_tree.get("token_boundaries", [])
    obj.snapped_cues = [
      SnappedCuePoint.load_from_cache(cue) for cue in cached_tree.get("snapped_cues", [])
    ]
    obj.tokens = [Token.from_dict(token) for token in cached_tree.get("tokens", [])]
    
    # Safely bind remaining pipeline settings or variables
    vocal_features = cached_tree.get("vocal_features", {})
    obj.vocal_density_threshold = vocal_features.get("vocal_density_threshold")
    obj.vocal_confidence_steepness = vocal_features.get("vocal_confidence_steepness")
    obj.global_vocal_features = vocal_features
    obj.bpm = cached_tree.get("bpm")
    
    # 4. Return the fully hydrated instance object shell
    return obj
  def LLM_representation(self):
     return{
        "name": self.name,
        "author": self.author,
        "duration_ms": self.duration_ms,
        "token_size_beats": 16, # TODO change into "self.dsp_manager.beats_per_token", now it's not possible because of the quick and dirty load_from_cache method, refactor this later by splitting the constructors and the pipeline execution. this is just to have something working for the prototype
        # TODO global dsp features. (currently in profiler.py)
        "key": self.key,
        "bpm": self.bpm,
        "tokens": [token.LLM_representation() for token in self.tokens]
     }
  
  @property
  def LLM_lyrics_profile(self) -> dict:
    """
    Computes global lyric tracking metrics and synthesizes a trust score
    by inspecting confidence boundaries across all text tokens.
    """
    min_scores = []
    median_scores = []
    
    # Safely iterate through our tokens array (handles both objects and namespaces)
    tokens = self.tokens
    for token in tokens:
      tokens = self.tokens
      full_text = ""
      for token in tokens:
        if token.lyrics != []:
          for word in token.lyrics["words"]:
            full_text += word["word"] + " "

          min_scores.append(token.lyrics["min_score"])
          median_scores.append(token.lyrics["median_score"])  

    # Compute global averages to determine our reliability threshold
    avg_min = sum(min_scores) / len(min_scores)
    avg_med = sum(median_scores) / len(median_scores)

    # Apply your system constants to establish the trust classification
    if avg_med >= HIGH_MEDIAN_SCORE and avg_min >= HIGH_MIN_SCORE:
        reliability_tag = "HIGH"
    elif avg_med < LOW_MEDIAN_SCORE or avg_min < LOW_MIN_SCORE:
        reliability_tag = "LOW (Hallucination Risk)"
    else:
        reliability_tag = "MEDIUM"

    return {
      "text": full_text,
      "reliability": reliability_tag,
      "metrics": {
        "avg_median": round(avg_med, 2),
        "avg_min": round(avg_min, 2)
      }
    }
    


    



if __name__ == "__main__":

  filename = sys.argv[1]
  repo_root = Path(__file__).resolve().parent.parent.parent
  audio_path = repo_root / "data" / "raw_audio" / filename
  track = Track(path = str(audio_path), name = filename, author = "unknown")
  print("\n\n\n\n############ BEST ##############")
  print("Best phase offset (in beats):", track.best_beat_phase_offset)
  print("Mean score for best phase offset:", track.mean_cue_score)
  print("Token boundaries (ms):", track.token_boundaries)
  print("Snapped Cue Points:")
  for snapped_cue in track.snapped_cues:
      print(snapped_cue.json())




  print("\n\n\n\n############ FULL TRACK JSON ##############")
  # print(json.dumps(track.json(), indent = 4))

  track_folder = Path(filename).stem
  full_json_path = repo_root / "data" / "json_db" / track_folder / "full.json"
  full_json_path.parent.mkdir(parents=True, exist_ok=True)
  with open(full_json_path, "w") as f:
      json.dump(track.json(), f, indent=4)
  print(f"Full track JSON saved to {full_json_path}")

  

  print("\n\n\n\n############ LLM REPRESENTATION ##############")
  # print(json.dumps(track.LLM_representation(), indent = 4))

  LLM_json_path = repo_root / "data" / "json_db" / track_folder / "LLM.json"
  with open(LLM_json_path, "w") as f:
      json.dump(track.LLM_representation(), f, indent=4)
  print(f"LLM representation JSON saved to {LLM_json_path}")

  print("\n\n\n\n############ TOKEN AUDIO SLICE TEST ##############")
  for token in track.tokens:
      token_audio_path = track.generate_token_audio(token.index)
      print(f"Generated audio slice for token {token.index} at {token_audio_path}")