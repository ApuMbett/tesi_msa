from beatnet_wrapper import BeatNetWrapper
from cue_point import RawCuePoint, SnappedCuePoint
import json
class Token: 
  def __init__(self, index, start_time_ms, end_time_ms, is_mixable, dsp_features, start_cue = None, end_cue = None):
    self.index = index
    self.start_time_ms = start_time_ms
    self.end_time_ms = end_time_ms
    self.is_mixable = is_mixable
    self.start_cue = start_cue
    self.end_cue = end_cue

    self.caption = None # TODO this will be generated later by the captioning model, it's not important for now
    self.dsp_features = dsp_features # TODO this will hold the dsp features for the token, such as bpm, key, energy, etc. we can compute these later using the beatnet wrapper and other tools. it's not important for now
  
  def get_audio_segment(self):
    # TODO this will return the audio segment corresponding to the token, pydub to do this. it's not important for now
    pass
  
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
    }
class Track:
  def __init__(self, path, name, author, beats_per_token = 16, structural_penalty_weight = 0.5):
    self.path = path
    self.name = name
    self.author = author

    self.bn = BeatNetWrapper(audio_path = path, beats_per_token = beats_per_token)
    self.raw_cue_points = self._compute_raw_cue_points()

    # self.dsp_features["bpm"] = self.bn.compute_global_bpm()

    # when we compute the token boundaries we find the phase offset that maximizes the score. the cue points are snapped to downbeats in a way that the mean score is maximized.
    self.best_phase_offset = None
    self.mean_score = None
    self.token_boundaries = []
    self.snapped_cues = []

    self._compute_snapped_cue_points(structural_penalty_weight = structural_penalty_weight)
    self.duration_ms = self.bn.beats[-1][0]


    self.tokens = self._build_token_map()


  def _compute_raw_cue_points(self) -> list[RawCuePoint]:
    # TODO: call cue-detr, we just read it from a json file for now.
    #debug: print full path 
    import os 
    print("Reading cue points from:", os.path.abspath("../../../cue-detr/tracks/_cue_points.json"))
    with open("../externals/cue-detr/tracks/_cue_points.json", "r") as f:

      data = json.load(f)
      shotmedown_cues = data[self.name] #! not exactly but now we just assume that name includes the extension

      # note: the cue points in the json file are in seconds, we need to convert them to milliseconds for the RawCuePoint class and i don't know if this is the right precision 
      return [RawCuePoint(time_ms = int(cue["time"]*1000), score = cue["score"]) for cue in shotmedown_cues]
    

  def _compute_snapped_cue_points(self, structural_penalty_weight = 0.5) -> list[SnappedCuePoint]:
    for i in range(self.bn.beats_per_token//4):
        snapped_cues = [SnappedCuePoint(raw_cue, self.bn, downbeat_offset_to_skip = i, structural_penalty_weight = structural_penalty_weight) for raw_cue in self.raw_cue_points]
        #! ^ this will be refactored later, snappedcuepoint should not take the whole beat grid as an argument now that the track class exists 

        # maximize the mean score across all snapped cues
        mean_score = sum(cue.score for cue in snapped_cues) / len(snapped_cues)
        if self.mean_score is None or mean_score > self.mean_score:
            self.mean_score = mean_score
            self.best_phase_offset = i
            self.snapped_cues = snapped_cues
            self.token_boundaries = self.bn.get_semantic_audio_token_boundaries(downbeat_offset_to_skip = i)
            #!^ problem: token boundaries now include tail and intro pad, this was necessary for completeness. now we can move this to the track class, this is necessary because (see blablabla) we have the micro intro pad that has 8000 bpm and so the qer is very high. even though it's few ms 
        
        print(f"\n\n\n\n######## Phase offset {i}: mean score={mean_score} ########") 
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

        # TODO DSP features
        dsp_features = {}
        if is_mixable:
          # compute the bpm only if it's not a pad 
          dsp_features["BPM"] = self.bn._compute_token_bpm(start_time_ms, end_time_ms)
        # TODO captioning 
        token = Token(i, start_time_ms, end_time_ms, is_mixable, dsp_features, start_cue = start_cue, end_cue = end_cue)
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



if __name__ == "__main__":

  filename = "shotmedown.mp3"
  track = Track(path = "../../data/raw_audio/" + filename, name = filename, author = "unknown")
  print("\n\n\n\n############ BEST ##############")
  print("Best phase offset (in beats):", track.best_phase_offset)
  print("Mean score for best phase offset:", track.mean_score)
  print("Token boundaries (ms):", track.token_boundaries)
  print("Snapped Cue Points:")
  for snapped_cue in track.snapped_cues:
      print(snapped_cue.json())

  print("\n\n\n\n############ TOKENS ##############")
  for token in track.tokens:
      print(json.dumps(token.json(), indent = 4))