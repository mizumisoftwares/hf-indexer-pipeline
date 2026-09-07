import os
import glob
import subprocess
import pandas as pd
import polars as pl
from huggingface_hub import HfApi, create_repo

# Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")
TARGET_REPO = os.getenv("DEST_REPO") # e.g. "username/my-parquet-dataset"
EXTRACT_DIR = "./extracted_data"
PARQUET_DIR = "./parquet_output"

os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)

def extract_rar():
    print("--> Joining and Extracting multi-part RAR files...")
    # Finds the first part (e.g., part01.rar or part1.rar)
    rar_parts = sorted(glob.glob("rar_inputs/*.rar"))
    if not rar_parts:
        raise FileNotFoundError("No RAR files found in rar_inputs/")
    
    # Run unrar on the primary file (unrar handles multi-part sequentially)
    cmd = f"unrar x -o+ {rar_parts[0]} {EXTRACT_DIR}/"
    subprocess.run(cmd, shell=True, check=True)
    print("--> Unrar completed successfully.")

def convert_to_parquet():
    print("--> Converting extracted files to Parquet...")
    files = glob.glob(f"{EXTRACT_DIR}/**/*", recursive=True)
    
    idx = 0
    for file_path in files:
        if os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            output_parquet = os.path.join(PARQUET_DIR, f"data_part_{idx}.parquet")
            
            try:
                if ext in ['.csv', '.tsv']:
                    # Polars fast engine for CSV -> Parquet
                    df = pl.read_csv(file_path, ignore_errors=True)
                    df.write_parquet(output_parquet, compression="zstd")
                    idx += 1
                elif ext in ['.json', '.jsonl']:
                    df = pl.read_json(file_path)
                    df.write_parquet(output_parquet, compression="zstd")
                    idx += 1
            except Exception as e:
                print(f"Skipping or error processing {file_path}: {e}")

def upload_to_hf():
    print("--> Uploading Parquet dataset to Hugging Face Hub...")
    api = HfApi()
    
    # Ensure dataset repository exists
    create_repo(repo_id=DEST_REPO, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
    
    # Upload whole parquet folder
    api.upload_folder(
        folder_path=PARQUET_DIR,
        repo_id=TARGET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        path_in_repo="data/"
    )
    print("--> Upload Complete!")

if __name__ == "__main__":
    extract_rar()
    convert_to_parquet()
    upload_to_hf()
