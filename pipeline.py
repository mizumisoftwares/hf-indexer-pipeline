import os
import glob
import subprocess
import duckdb
import shutil
from huggingface_hub import hf_hub_download, HfApi, login

HF_TOKEN = os.getenv("HF_TOKEN")
DEST_REPO = os.getenv("DEST_REPO", "sarveshmgkvp/extracted-foab-parquet")
START_PART = int(os.getenv("START_PART", "1"))
END_PART = int(os.getenv("END_PART", "1"))

# Corrected typo: added 's' to Breaches
SOURCE_REPO = "darrifylive/Father-of-All-Breache-FOAB"

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set!")

login(token=HF_TOKEN)
api = HfApi()

api.create_repo(repo_id=DEST_REPO, repo_type="dataset", private=True, exist_ok=True)

print(f"=== Starting Processing Pipeline for Parts {START_PART} through {END_PART} ===")

part_dir = "./temp_rar_chunk"
extract_dir = "./temp_extract_chunk"
output_dir = "./temp_out_chunk"

os.makedirs(part_dir, exist_ok=True)
os.makedirs(extract_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

part1_filename = "xpolite-emaildb.part-2-.part001.rar"

try:
    # 1. Download Header Part with explicit auth token
    print(f"Downloading Archive Header: {part1_filename}")
    hf_hub_download(
        repo_id=SOURCE_REPO,
        filename=part1_filename,
        repo_type="dataset",
        local_dir=part_dir,
        token=HF_TOKEN
    )
    
    # 2. Download target parts
    for part in range(START_PART, END_PART + 1):
        target_filename = f"xpolite-emaildb.part-2-.part{part:03d}.rar"
        if part != 1:
            print(f"Downloading Volume Part {part}: {target_filename}")
            try:
                hf_hub_download(
                    repo_id=SOURCE_REPO,
                    filename=target_filename,
                    repo_type="dataset",
                    local_dir=part_dir,
                    token=HF_TOKEN
                )
            except Exception as dl_err:
                print(f"Notice: Could not download {target_filename}: {dl_err}")

    # 3. Extract RAR volumes using 7-Zip
    part1_path = os.path.join(part_dir, part1_filename)
    print("\nExtracting archive volumes using 7-Zip...")
    cmd_res = subprocess.run(
        ["7z", "x", part1_path, f"-o{extract_dir}", "-y"],
        capture_output=True,
        text=True
    )
    print(f"7-Zip Execution Finished.")

    # 4. Recursively collect all extracted files
    extracted_files = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            extracted_files.append(os.path.join(root, file))

    print(f"\nFound {len(extracted_files)} extracted file(s).")

    # 5. Convert to Parquet
    for idx, filepath in enumerate(extracted_files):
        file_name = os.path.basename(filepath)
        if file_name.startswith('.') or file_name.endswith(('.duckdb', '.parquet')):
            continue
            
        pq_name = f"{output_dir}/file_{idx}.parquet"
        print(f"  [Processing]: {file_name}")
        
        try:
            query = f"""
                COPY (
                    SELECT * FROM read_csv('{filepath}', 
                        header=false, 
                        ignore_errors=true, 
                        all_varchar=true,
                        auto_detect=true)
                ) TO '{pq_name}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');
            """
            duckdb.execute(query)
            print(f"    -> Successfully created Parquet: {os.path.basename(pq_name)}")
            
        except Exception as err:
            print(f"    -> [Warning] Could not parse {file_name}: {err}")

    # 6. Upload output directory to Hugging Face
    output_files = glob.glob(f"{output_dir}/*")
    print(f"\nUploading {len(output_files)} parsed file(s) to Hugging Face...")
    
    if output_files:
        api.upload_folder(
            folder_path=output_dir,
            repo_id=DEST_REPO,
            repo_type="dataset",
            path_in_repo=f"batch_{START_PART}_to_{END_PART}",
            multi_commits=True
        )
        print("Upload finished successfully!")
    else:
        print("No output files generated to upload.")

finally:
    print("Cleaning up temporary directories...")
    shutil.rmtree(part_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
