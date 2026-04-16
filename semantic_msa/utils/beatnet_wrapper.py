from BeatNet.BeatNet import BeatNet
import numpy as np
from collections import Counter
class BeatNetWrapper:
  """
    Wrapper around BeatNet to extract beat grid information and convert it to a format suitable for our application. This class can be initialized with either raw BeatNet output or an audio file path, and provides methods to retrieve the full beat grid, downbeats, pixel boundaries for blocks of bars, and both global and local BPM estimates. 
  """
  def __init__(self, raw_beatnet_output = None, audio_path = None):
    if raw_beatnet_output is not None:
        self.beats = BeatNetWrapper._sanitize_raw_data(raw_beatnet_output)
    elif audio_path is not None:
        self.beats = BeatNetWrapper._compute_raw_data(audio_path)
    else:
        raise ValueError("Must provide either audio_path or raw_beatnet_output")

  @staticmethod
  def _compute_raw_data(audio_path):
    estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
    data = estimator.process(audio_path)  
    
    return BeatNetWrapper._sanitize_raw_data(data)

  @staticmethod
  def _sanitize_raw_data(raw_data):
      return [(int(time*1000), int(beat)) for time, beat in raw_data]
      # return [(time, int(beat)) for time, beat in raw_data]


  def get_full_grid(self):
    return self.beats
    
  def get_downbeats(self):
    return [time for time, beat in self.beats if beat == 1]


  def get_semantic_audio_token_boundaries(self, beats_per_token = 16):
    """
    Converts the beat grid into boundaries for semantic audio tokens, which represent blocks of bars. This method identifies the first downbeat to ensure phase-locked phrasing, slices the grid accordingly, and applies intro and tail padding to capture the full track duration. The resulting boundaries are crucial for structuring the track into meaningful segments for DJ mix preparation.
    """
    # 1. Find the index of the FIRST downbeat (where beat == 1 or 1.0) to ensure phase-locked 16-beat phrasing
    try:
        anchor_idx = next(i for i, (time, beat) in enumerate(self.beats) if beat == 1)
    except StopIteration:
        anchor_idx = 0  # Failsafe if no downbeats are found at all
        
    # 2. Slice the grid starting strictly from the Anchor
    boundaries = [time for time, beat in self.beats[anchor_idx::beats_per_token]]
    
    # 3. Intro Pad: Insert 0 only if the first downbeat isn't already at 0
    if boundaries and boundaries[0] > 0:
        boundaries.insert(0, 0)
        
    # 4. Tail Pad: Append the last beat only if it isn't already captured
    last_time = self.beats[-1][0]
    if boundaries and boundaries[-1] != last_time:
        boundaries.append(last_time)
        
    return boundaries

  
    
  def compute_global_bpm(self): 
      """Returns the median BPM of the entire track.

      This method computes the BPM for each 16-beat token and returns the most common BPM as the global estimate. This approach accounts for potential tempo variations across the track while providing a robust overall BPM estimate for mix preparation.
      """
      bpm_estimates = []
      token_boundaries = self.get_semantic_audio_token_boundaries()
      for i in range(len(token_boundaries)-1):
        start_time = token_boundaries[i]
        end_time = token_boundaries[i+1]
        # find the corresponding beat indices for the start and end times
        start_beat_index = next(j for j, (time, beat) in enumerate(bn.get_full_grid()) if time >= start_time)
        end_beat_index = next(j for j, (time, beat) in enumerate(bn.get_full_grid()) if time >= end_time)
        bpm = bn.compute_bpm(start_beat_index, end_beat_index)
        if bpm is not None:
          bpm_estimates.append(bpm)
      
      most_common = Counter(bpm_estimates).most_common(1)
      return float(most_common[0][0])


  def compute_bpm(self, start_beat_index, end_beat_index):
      """Returns the macro BPM for a specific segment by measuring total duration."""
      segment_beats = self.beats[start_beat_index:end_beat_index]
      num_beats = len(segment_beats)
      
      # We need at least 2 beats to measure an interval
      if num_beats < 2:
          return None  
          
      # Get the exact start and end times in milliseconds
      start_time_ms = segment_beats[0][0]
      end_time_ms = segment_beats[-1][0]
      duration_ms = end_time_ms - start_time_ms
      
      if duration_ms <= 0:
          return None
          
      # The number of intervals (spaces between beats) is num_beats - 1
      intervals = num_beats - 1
      
      # Formula: (intervals / duration_in_minutes)
      bpm = (intervals / (duration_ms / 60000))
      
      return round(bpm, 1) # Rounds to a clean decimal like 128.0

if __name__ == "__main__":
    print("started beatnet wrapper test")
    # resolve path 
    path = "../data/raw_audio/shotmedown.mp3"

    bn = BeatNetWrapper(audio_path = path)

    print("full grid:", bn.get_full_grid())
    print("downbeats:", bn.get_downbeats())
    token_boundaries = bn.get_semantic_audio_token_boundaries()
    print("token boundaries:", token_boundaries)
    print(len(bn.get_semantic_audio_token_boundaries()))

    print("global BPM:", bn.compute_global_bpm())
    print("BPM for each 16-beat token: ")
    bpms = []
    for i in range(len(token_boundaries)-1):
        start_time = token_boundaries[i]
        end_time = token_boundaries[i+1]
        # find the corresponding beat indices for the start and end times
        start_beat_index = next(j for j, (time, beat) in enumerate(bn.get_full_grid()) if time >= start_time)
        end_beat_index = next(j for j, (time, beat) in enumerate(bn.get_full_grid()) if time >= end_time)
        bpm = bn.compute_bpm(start_beat_index, end_beat_index)
        if bpm is not None: 
          bpms.append(bpm)
        print(f"Token {i}: BPM={bpm}")
    
    print("Median BPM across tokens:", np.median(bpms))
