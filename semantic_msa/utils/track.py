from beatnet_wrapper import BeatNetWrapper
from cue_point import RawCuePoint, SnappedCuePoint
import json

class Track:
  def __init__(self, path, name, author, beats_per_token = 16):
    self.path = path
    self.name = name
    self.author = author

    self.bn = BeatNetWrapper(audio_path = path, beats_per_token = beats_per_token)
    self.raw_cue_points = self._compute_raw_cue_points()

    # self.dsp_features["bpm"] = self.bn.compute_global_bpm()

    # when we compute the token boundaries we find the phase offset that minimizes the mean qer. the cue points are snapped to downbeats in a way that the mean qer is minimized.
    self.best_phase_offset = None
    self.mean_qer = None
    self.token_boundaries = []
    self.snapped_cues = []

    self._compute_snapped_cue_points()



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
    

  def _compute_snapped_cue_points(self) -> list[SnappedCuePoint]:
    for i in range(self.bn.beats_per_token//4):
        snapped_cues = [SnappedCuePoint(raw_cue, self.bn, downbeat_offset_to_skip = i) for raw_cue in self.raw_cue_points]
        #! ^ this will be refactored later, snappedcuepoint should not take the whole beat grid as an argument now that the track class exists 
        mean_qer = sum(cue.quantization_error_ratio for cue in snapped_cues) / len(snapped_cues)
        if self.mean_qer is None or mean_qer < self.mean_qer:
            self.mean_qer = mean_qer
            self.best_phase_offset = i
            self.snapped_cues = snapped_cues
            self.token_boundaries = self.bn.get_semantic_audio_token_boundaries(downbeat_offset_to_skip = i)
        
        print(f"\n\n\n\n######## Phase offset {i}: mean QER={mean_qer} ########") 
        print(F"token boundaries={self.token_boundaries}") 
        print("Snapped Cue Points for this phase offset:")
        for snapped_cue in snapped_cues:
            print(snapped_cue.json())
  

if __name__ == "__main__":
  import os 
  print("Testing Track class with audio file:", os.path.abspath("../data/raw_audio/shotmedown.mp3"))
  track = Track(path = "../../data/raw_audio/shotmedown.mp3", name = "shotmedown.mp3", author = "unknown")
  track._compute_snapped_cue_points()

  print("\n\n\n\n############ BEST ##############")
  print("Best phase offset (in beats):", track.best_phase_offset)
  print("Mean Quantization Error Ratio for best phase offset:", track.mean_qer)
  print("Token boundaries (ms):", track.token_boundaries)
  print("Snapped Cue Points:")
  for snapped_cue in track.snapped_cues:
      print(snapped_cue.json())