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

    self.caption = None # TODO this will be generated later by the captioning model, it's not important for now
    self.dsp_features = dsp_features # TODO this will hold the dsp features for the token, such as bpm, key, energy, etc. we can compute these later using the beatnet wrapper and other tools. it's not important for now
    self.lyrics = lyrics

  
  def get_audio_segment(self):
    # TODO this will return the audio segment corresponding to the token, pydub to do this. it's not important for now
    pass
  

  # TODO
  def _get_DSP_features_LLM_representation(self):
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
      "lyrics": self.lyrics
    }
  
  def LLM_representation(self):
    return {
      "index": self.index,
      "start_time_ms": self.start_time_ms,
      "end_time_ms": self.end_time_ms,
      "is_mixable": self.is_mixable,
      "start_cue": self.start_cue.json() if self.start_cue else None,
      "end_cue": self.end_cue.json() if self.end_cue else None,
      "caption": self.caption,
      "dsp_features": self.dsp_features,
      "lyrics": self._get_lyrics_LLM_representation() if self.lyrics else None
    }
class Track:
  def __init__(self, path, name, author, beats_per_token = 16, structural_penalty_weight = 0.5):
    self.path = path
    self.name = name
    self.author = author

    self.dsp_manager = DSPManager(audio_path = path, beats_per_token = beats_per_token)
    self.raw_cue_points = self._compute_raw_cue_points()

    # self.dsp_features["bpm"] = self.dsp_manager.compute_global_bpm()

    # when we compute the token boundaries we find the phase offset that maximizes the score. the cue points are snapped to downbeats in a way that the mean score is maximized.
    self.best_beat_phase_offset = None
    self.mean_cue_score = None
    self.token_boundaries = []
    self.snapped_cues = []

    self._compute_snapped_cue_points(structural_penalty_weight = structural_penalty_weight)
    self.duration_ms = self.dsp_manager.beats[-1][0]

    self.lyrics = self.dsp_manager.get_lyrics()
    
    self.vocal_density_threshold = self.dsp_manager._get_vocal_threshold()
    self.vocal_confidence_steepness = self.dsp_manager._get_vocal_confidence_steepness()
    self.tokens = self._build_token_map()

    self.key = self.dsp_manager.compute_camelot_key(0, self.duration_ms)
    self.bpm = None #TODO PLACEHOLDER


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

  def json(self):
    return {
      "path": self.path,
      "name": self.name,
      "author": self.author,
      "key": self.key,
      "duration_ms": self.duration_ms,
      "best_beat_phase_offset": self.best_beat_phase_offset,
      "mean_cue_score": self.mean_cue_score,
      "vocal_features": self.global_vocal_features,
      "token_boundaries": self.token_boundaries,
      "snapped_cues": [cue.json() for cue in self.snapped_cues],
      "tokens": [token.json() for token in self.tokens]
    }
  
  def LLM_representation(self):
     return{
        "name": self.name,
        "author": self.author,
        "duration_ms": self.duration_ms,
        "key": self.key,
        "bpm": self.bpm,
        "tokens": [token.LLM_representation() for token in self.tokens]
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

  print("\n\n\n\n############ TOKENS ##############")
  for token in track.tokens:
      print(json.dumps(token.json(), indent = 4))



  print("\n\n\n\n############ FULL TRACK JSON ##############")
  print(json.dumps(track.json(), indent = 4))

  track_folder = Path(filename).stem
  full_json_path = repo_root / "data" / "json_db" / track_folder / "full.json"
  full_json_path.parent.mkdir(parents=True, exist_ok=True)
  with open(full_json_path, "w") as f:
      json.dump(track.json(), f, indent=4)
  print(f"Full track JSON saved to {full_json_path}")

  

  print("\n\n\n\n############ LLM REPRESENTATION ##############")
  print(json.dumps(track.LLM_representation(), indent = 4))

  LLM_json_path = repo_root / "data" / "json_db" / track_folder / "LLM.json"
  with open(LLM_json_path, "w") as f:
      json.dump(track.LLM_representation(), f, indent=4)
  print(f"LLM representation JSON saved to {LLM_json_path}")