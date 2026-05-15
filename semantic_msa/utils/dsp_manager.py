from BeatNet.BeatNet import BeatNet
import numpy as np
from collections import Counter
import librosa
import os
import subprocess
import json

AUDIO_DIR = "../data/raw_audio/"
class DSPManager:
  """
    Wrapper around BeatNet to extract beat grid information and convert it to a format suitable for our application. This class can be initialized with either raw BeatNet output or an audio file path, and provides methods to retrieve the full beat grid, downbeats, pixel boundaries for blocks of bars, and both global and local BPM estimates. 
  """
  def __init__(self, audio_path, beats_per_token = 16):
    self.beats_per_token = beats_per_token
    self.beats = DSPManager._compute_raw_data(audio_path)

    # loading the track into memory using librosa, useful for low level DSP features computing 
    self.sr = 22050  # Standard sampling rate for audio processing
    self.y = None 
    self.y, self.sr = librosa.load(audio_path, sr=self.sr)
    self.audio_path = audio_path
    self.track_name = audio_path.split("/")[-1].split(".")[0]    

  @staticmethod
  def _compute_raw_data(audio_path):
    estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
    data = estimator.process(audio_path)  
    
    return DSPManager._sanitize_raw_data(data)

  @staticmethod
  def _sanitize_raw_data(raw_data):
      return [(int(time*1000), int(beat)) for time, beat in raw_data]
      # return [(time, int(beat)) for time, beat in raw_data]
  

  def get_full_grid(self):
    return self.beats
    
  def get_downbeats(self):
    return [time for time, beat in self.beats if beat == 1]


  def get_semantic_audio_token_boundaries(self, downbeat_offset_to_skip = 0):
    """
    Converts the beat grid into boundaries for semantic audio tokens, which represent blocks of bars. This method identifies the first downbeat after downbeat_offset_to_skip to ensure phase-locked phrasing and slices the grid accordingly.
    """
    # 1. Find the index of the FIRST downbeat (anchor) (where beat == 1 or 1.0) to ensure phase-locked 16-beat phrasing
    # Find the indices of ALL downbeats (where beat == 1)
    downbeat_indices = [i for i, (time, beat) in enumerate(self.beats) if beat == 1]
    
    if not downbeat_indices:
        anchor_idx = 0  # Failsafe
    else:
        # Safely get the Nth downbeat (wrap around if offset is somehow too high)
        safe_offset = min(downbeat_offset_to_skip, len(downbeat_indices) - 1)
        anchor_idx = downbeat_indices[safe_offset]  

    # 2. Slice the grid starting strictly from the Anchor
    boundaries = [time for time, beat in self.beats[anchor_idx::self.beats_per_token]]
        
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
        bpm = self._compute_token_bpm(start_time, end_time)
        if bpm is not None:
          bpm_estimates.append(bpm)
      
      most_common = Counter(bpm_estimates).most_common(1)
      return float(most_common[0][0])

  def _compute_token_bpm(self, start_time_ms, end_time_ms):
        """Returns the BPM for a specific token defined by start_time and end_time.

            for performance reasons, this helper method computes the BPM for a token which we know is self.beats_per_token beats long so we can directly calculate the BPM using the duration of the token, without needing to count the number of beats within it. This is more efficient than the general compute_bpm method when we are specifically working with fixed-length tokens.
        """
        # find the corresponding beat indices for the start and end times
        duration_ms = end_time_ms - start_time_ms
      
        if duration_ms <= 0:
          return None
          
        bpm = (self.beats_per_token * 60000) / duration_ms
        return round(bpm, 1)

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

  def compute_low_level_dsp_features(self, start_time_ms, end_time_ms):
    # TODO this is temporary but i just need it to see if it works, i will investigate later for better features for our purposes 
    # 1. Convert milliseconds to array indices
    start_sample = int(start_time_ms * self.sr / 1000)
    end_sample = int(end_time_ms * self.sr / 1000)
    
    # 2. Slice the audio array for this specific token
    y_token = self.y[start_sample:end_sample]
    
    # --- THE REST OF THE PERPLEXITY SOTA MATH ---
    S = np.abs(librosa.stft(y_token))
    rms_env = librosa.feature.rms(S=S)[0]
    total_rms_mean = np.mean(rms_env) + 1e-6
    
    onset_env = librosa.onset.onset_strength(y=y_token, sr=self.sr, S=S)
    
    freqs = librosa.fft_frequencies(sr=self.sr)
    bass_mask = freqs < 250
    mid_mask = (freqs >= 250) & (freqs < 4000)
    treb_mask = freqs >= 4000


    bass_rms = np.sqrt(np.mean(S[bass_mask, :]**2, axis=0))
    mid_rms = np.sqrt(np.mean(S[mid_mask, :]**2, axis=0))
    treb_rms = np.sqrt(np.mean(S[treb_mask, :]**2, axis=0))

    raw_bass = np.mean(np.sqrt(np.mean(S[bass_mask, :]**2, axis=0)))
    raw_mid = np.mean(np.sqrt(np.mean(S[mid_mask, :]**2, axis=0)))
    raw_treb = np.mean(np.sqrt(np.mean(S[treb_mask, :]**2, axis=0)))
    # normalize between 0 and 1
    total_band_energy = raw_bass + raw_mid + raw_treb + 1e-6
    
    btr = raw_bass / total_band_energy
    mtr = raw_mid / total_band_energy
    ttr = raw_treb / total_band_energy

    nyquist = self.sr / 2
    sc = np.mean(librosa.feature.spectral_centroid(S=S)) / nyquist
    sf = np.mean(librosa.feature.spectral_flatness(S=S))
    msf = np.mean(np.maximum(0, np.diff(S, axis=1))**2)
    
    roe = np.mean(onset_env)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
    ocn = len(onsets)
    ev = np.var(rms_env)
    
    S_harmonic, S_percussive = librosa.decompose.hpss(S)
    her = np.mean(librosa.feature.rms(S=S_harmonic)) / total_rms_mean

    return {
        "ROE": round(float(roe), 4),
        "BTR": round(float(btr), 4),
        "MTR": round(float(mtr), 4),
        "TTR": round(float(ttr), 4),
        "SC": round(float(sc), 4),
        "MSF": round(float(msf), 4),
        "SF": round(float(sf), 4),
        "OCN": int(ocn),
        "EV": round(float(ev), 4),
        "HER": round(float(her), 4)
    }

  def split_song_tracks(self):
    # this method splits the song into vocal and instrumental tracks using Demucs. 
    if not self.has_splitted_tracks():
      # run demucs and save the vocal and instrumental tracks in memory for later use.
      print(f"[Demucs] Extracting vocals for {self.audio_path}...")

      command = ["demucs", "--two-stems=vocals", "-o", AUDIO_DIR, self.audio_path]
      subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
  
  def has_splitted_tracks(self):
    # this method checks if the song has already been splitted into vocal and instrumental tracks, to avoid doing it multiple times. 
    expected_vocal_path = os.path.join(AUDIO_DIR, "htdemucs", self.track_name, "vocals.wav")
    
    if os.path.exists(expected_vocal_path):
        return True
    else:
      return False

  #TODO don't know if this is the right place for these methods but for now it's easier to implement them here since they are related to the DSP features of the track, we can refactor later if needed.
  def get_lyrics(self):
    # then runs RMS-VAD on the vocal track of the song and uses WhisperX to get the lyrics. 
    # The output is the timestamps of the lyrics, word by word.

    # 1. split the song into vocals and instrumental using demucs
    self.split_song_tracks()

    # 2. run WhisperX on the vocal track with custom VAD (RMS-VAD, Syed et al) to get the lyrics with timestamps. 
    # 2. Load the main Whisper into memory 
    # TODO (will inject RMS-VAD later)
    # i run it as a subprocess because it's easier to manage the dependencies 
    output_json_path = f"../data/json_db/{self.track_name}/_whisper_output.json"
    if not os.path.exists(output_json_path):
      vocal_path = os.path.join(AUDIO_DIR, "htdemucs", self.track_name, "vocals.wav")
      whisper_env_python = "../whisper_engine/.venv/bin/python"  # Path to the Python executable in the whisper environment
      command = [whisper_env_python, "../whisper_engine/whisper_wrapper.py", vocal_path]
      subprocess.run(command, check=True, )
      # Read the JSON file left behind by the bridge script
      print("Transcription complete. Ingesting timestamp data...")
      if not os.path.exists(output_json_path):
          raise FileNotFoundError("WhisperX finished, but no JSON output was found!")

    with open(output_json_path, "r", encoding="utf-8") as f:
        lyrics_data = json.load(f)

    # 4. (Optional but recommended) Clean up the JSON file so it doesn't clutter your folder
    # os.remove(output_json_path)

    return lyrics_data

    

    

    pass 
  def compute_vocal_features(self, start_time_ms, end_time_ms):
    # this method calculates vocal_confidence, vocal_presence, start_BVR and end_BVR for the interval. 
    # it does this on the vocal_track of the song  

    pass
if __name__ == "__main__":
    print("started beatnet wrapper test")
    # resolve path 
    path = "../data/raw_audio/shotmedown.mp3"

    bn = DSPManager(audio_path = path)

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

    print("LYRICS TEST")
    bn.split_song_tracks()

    lyrics = bn.get_lyrics()
    print(lyrics)
