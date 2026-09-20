import pandas as pd 
from huggingface_hub import hf_hub_download
import os
from tqdm import tqdm
import subprocess
from wakepy import keep
import argparse

def download_songs(dataset_path):
    # if songs not already downloaded
    # get dataset 
    df = pd.read_csv(dataset_path)
    # download songs 

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Total Progress", position=1, leave=True):
        hf_hub_download(
            repo_id="ASLP-lab/SongFormBench",
            repo_type="dataset",
            filename=row["audio_path"],
            local_dir="./downloaded_data"
        )

def run_pipeline(dataset_csv_path, download_dir, workers):
    print("\nStarting Semantic MSA Batch Pipeline...")
    
    # Path to semantic_msa directory (one level up from eval)
    semantic_msa_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "semantic_msa"))
    
    # We must run it with uv run --no-sync to protect the whisperx dependency
    command = [
        "uv", "run", "--no-sync", "src/semantic_msa/batch_process.py",
        "--dataset", os.path.abspath(dataset_csv_path),
        "--download-dir", os.path.abspath(download_dir),
        "--workers", str(workers)
    ]
    
    # Isolate outputs to eval/eval_processed
    eval_processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "eval_processed"))
    env = os.environ.copy()
    env["SEMANTIC_MSA_DATASET_ROOT"] = eval_processed_dir
    
    # =================================================================================
    # CRITICAL ENVIRONMENT SANITIZATION
    # ---------------------------------------------------------------------------------
    # When this script is run via `uv run main.py`, the active environment leaks down 
    # to the subprocess via VIRTUAL_ENV and PATH. Furthermore, because the parent folder
    # (`tesi`) is configured as a `uv` workspace, `uv run` natively ignores local `.venv`
    # folders and forcefully resolves to the root workspace `.venv`.
    #
    # We must explicitly "blind" the child process to the parent environment and 
    # aggressively force it to use `semantic_msa/.venv` where the manual PyTorch 
    # cu128 and WhisperX hack is installed.
    # =================================================================================
    
    # 1. Scrub parent virtual environment variables
    if "VIRTUAL_ENV" in env:
        del env["VIRTUAL_ENV"]
    if "UV_RUN_RECURSION_DEPTH" in env:
        del env["UV_RUN_RECURSION_DEPTH"]
        
    # 2. Hard-override the uv workspace behavior to target the local semantic_msa env
    env["UV_PROJECT_ENVIRONMENT"] = os.path.join(semantic_msa_dir, ".venv")
        
    # 3. Scrub the active virtual environment bin from PATH so uv doesn't fall back to it
    path = env.get("PATH", "")
    paths = path.split(os.pathsep)
    paths = [p for p in paths if not p.endswith(".venv/bin")]
    env["PATH"] = os.pathsep.join(paths)
    
    try:
        print("\nAttempting to prevent system sleep during batch processing...")
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with keep.running():
                if w:
                    print("  Note: Sleep prevention failed (expected if you are on WSL). Please make sure Windows is set to stay awake!")
                subprocess.run(command, cwd=semantic_msa_dir, env=env, check=True)
        print("\nPipeline completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed with exit code {e.returncode}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Semantic MSA pipeline")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers for batch processing")
    args = parser.parse_args()

    dataset_path = "./dataset_loaders/sampled_eval_dataset_enriched.csv"
    download_dir = "./downloaded_data"
    
    download_songs(dataset_path)
    run_pipeline(dataset_path, download_dir, args.workers)

if __name__ == "__main__":
    main()
