from semantic_msa.domain.workspace import TrackWorkspace
from semantic_msa.utils.track_analyzer import TrackAnalyzer
from semantic_msa.utils.logger import logger, console
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Process track name and audio path.")
    parser.add_argument("--name", type=str, required=True, help="Track name")
    parser.add_argument("--author", type=str, required=True, help="Track author")
    parser.add_argument("--audio-path", type=str, required=True, help="Track audio path")

    args = parser.parse_args()

    track_ws = TrackWorkspace(args.name, Path(args.audio_path))
    
    # Set up file logger for this specific run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add_file_handler(track_ws.track_dir / "logs", run_id)
    
    logger.step(f"Starting analysis for {args.author} - {args.name}")

    try:
        extractor = TrackAnalyzer(track_ws, args.author)
        track = extractor.extract()
        logger.success("Analysis complete.")
    except Exception as e:
        logger.error(f"Pipeline crashed during execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
   
    main()
