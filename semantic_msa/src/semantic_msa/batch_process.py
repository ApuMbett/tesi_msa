import argparse
import pandas as pd
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn

# CRITICAL: Force matplotlib to headless mode before any internal modules load it.
# This prevents Tcl/Tk (tkinter) from crashing with "RuntimeError: main thread is not in main loop" 
# when analyzing tracks concurrently in multiple threads.
import matplotlib
matplotlib.use("Agg")

from semantic_msa.utils.logger import logger, console
from semantic_msa.main import analyze_single_track
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_row(row, args):
    title = str(row["Title"])
    artist = str(row["Artist"])

    # Resolve audio path
    audio_rel_path = row["audio_path"]
    audio_abs_path = Path(args.download_dir) / audio_rel_path
    
    if not audio_abs_path.exists():
        logger.error(f"Audio file not found: {audio_abs_path}")
        return

    from semantic_msa.domain.workspace import TrackWorkspace
    workspace = TrackWorkspace(title, audio_abs_path)
    if (workspace.output_dir / "LLM.json").exists():
        logger.cache(f"Track '{title}' already fully processed. Skipping.")
        return

    try:
        analyze_single_track(title, artist, audio_abs_path)
    except Exception as e:
        logger.error(f"Failed to process {title}: {str(e)}", exc_info=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to enriched CSV dataset")
    parser.add_argument("--download-dir", type=str, required=True, help="Base directory where audio is downloaded")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (threads)")
    args = parser.parse_args()

    df_dataset = pd.read_csv(args.dataset)

    logger.info(f"Starting batch processing of {len(df_dataset)} tracks with {args.workers} workers...")
    
    completed = 0
    total = len(df_dataset)

    import os
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(process_row, row, args): row 
        for _, row in df_dataset.iterrows()
    }
    
    try:
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result() # Propagate any unexpected exceptions
            except Exception as e:
                logger.error(f"Task for {row['Title']} raised an unhandled exception: {e}")
            finally:
                completed += 1
                logger.info(f"==================================================")
                logger.info(f"BATCH PROGRESS: {completed}/{total} tracks completed")
                logger.info(f"==================================================")
    except KeyboardInterrupt:
        logger.warning("\nPipeline forcefully stopped by user (Ctrl+C). Exiting immediately...")
        os._exit(130)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

if __name__ == "__main__":
    main()
