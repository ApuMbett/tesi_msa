from BeatNet.BeatNet import BeatNet
import os
from typing import Any
from loguru import logger

def extract_beat_grid(audio_path: str):
  logger.info("Starting beat extraction: {}", audio_path)
  estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
  grid = estimator.process(audio_path)
  logger.info("Beat extraction completed. Rows found: {}", len(grid) if grid is not None else 0)
  return grid

def grid_to_json(grid, track_name: str = "blablabla", period_size_beats: int = 16) -> dict[str, Any]:
  logger.info(
    "Converting grid to JSON payload for track='{}' with period_size_beats={}",
    track_name,
    period_size_beats,
  )

  # BeatNet returns shape (num_beats, 2): [beat_time, downbeat_flag].
  beats = [list(map(float, row)) for row in grid]

  if not beats:
    logger.warning("No beats found in grid. Returning empty payload.")
    return {
      "track_name": track_name,
      "beats": [],
      "period_size_beats": period_size_beats,
      "total_blocks_found": 0,
      "block_boundaries": [],
    }

  # Find first downbeat (flag ~= 1). Fallback to first beat if no downbeat is marked.
  first_downbeat_idx = next(
    (idx for idx, beat in enumerate(beats) if len(beat) > 1 and beat[1] >= 0.5),
    0,
  )

  # Boundaries are one every period_size_beats starting from the first downbeat.
  block_boundaries = [beats[idx][0] for idx in range(first_downbeat_idx, len(beats), period_size_beats)]

  logger.info(
    "JSON payload ready: beats={}, first_downbeat_idx={}, total_blocks_found={}",
    len(beats),
    first_downbeat_idx,
    len(block_boundaries),
  )

  return {
    "track_name": track_name,
    "beats": beats,
    "period_size_beats": period_size_beats,
    "total_blocks_found": len(block_boundaries),
    "block_boundaries": block_boundaries,
  }

if __name__ == "__main__":
  # get arguments from argparse import ArgumentParser
  import argparse
  import json
  import sys

  logger.remove()
  logger.add(sys.stderr, level="INFO")

  parser = argparse.ArgumentParser(description='Extract beat grid from audio file')
  parser.add_argument('audio_path', type=str, help='Path to audio file')
  args = parser.parse_args()

  grid = extract_beat_grid(args.audio_path)
  payload = grid_to_json(grid, track_name=os.path.basename(args.audio_path))
  logger.info("Printing JSON payload to stdout.")
  print(json.dumps(payload, indent=2))

  
