import glob
import os
import shutil
import subprocess
import duckdb
from huggingface_hub import HfApi, hf_hub_download, login

HF_TOKEN = os.getenv("HF_TOKEN")
DEST_REPO = os.getenv("DEST_REPO", "sarveshmgkvp/extracted-foab-parquet")
SOURCE_REPO = "darrifylive/Father-of-All-Breaches-FOAB"

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set!")

login(token=HF_TOKEN)
api = HfApi()

api.create_repo(repo_id=DEST_REPO, repo_type="dataset", private=True, exist_ok=True)

# Prompt user for specific target filename dynamically
target_filename = input("Enter the full part filename (e.g., xpolite-emaildb.part-2-.part005.rar): ").strip()

if not target_filename:
    raise ValueError("Filename cannot be empty!")

print(f"=== Starting Processing Pipeline for File: {target_filename} ===")

part_dir = os.path.abspath("./temp_rar_chunk")
extract_dir = os.path.abspath("./temp_extract_chunk")
output_dir = os.path.abspath("./temp_out_chunk")

os.makedirs(part_dir, exist_ok=True)
os.makedirs(extract_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

part1_filename = "xpolite-emaildb.part-2-.part001.rar"

try:
    # 1. Download Header Part
    print(f"Downloading Archive Header: {part1_filename}")
    hf_hub_download(
        repo_id=SOURCE_REPO,
        filename=part1_filename,
        repo_type="dataset",
        local_dir=part_dir,
        token=HF_TOKEN,
    )

    # 2. Download requested single target file if distinct from header
    if target_filename != part1_filename:
        print(f"Downloading Requested Part: {target_filename}")
        hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=target_filename,
            repo_type="dataset",
            local_dir=part_dir,
            token=HF_TOKEN,
        )

    # 3. Extract RAR volume using 7-Zip
    part1_path = os.path.join(part_dir, part1_filename)
    print("\nExtracting archive volume using 7-Zip...")
    cmd_res = subprocess.run(
        ["7z", "x", part1_path, f"-o{extract_dir}", "-y"],
        capture_output=True,
        text=True,
        check=False,
    )

    if cmd_res.returncode != 0:
        print(f"[Warning] 7-Zip exited with code {cmd_res.returncode}: {cmd_res.stderr}")

    # 4. Recursively collect extracted files
    extracted_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(extract_dir)
        for file in files
    ]
    print(f"\nFound {len(extracted_files)} extracted file(s).")

    # 5. Convert to Parquet
    con = duckdb.connect()
    try:
        for idx, filepath in enumerate(extracted_files):
            file_name = os.path.basename(filepath)
            if file_name.startswith(".") or file_name.endswith((".duckdb", ".parquet")):
                continue

            pq_path = os.path.join(output_dir, f"file_{idx}.parquet")
            print(f"  [Processing]: {file_name}")

            try:
                query = """
                    COPY (
                        SELECT * FROM read_csv(?, 
                            header=false, 
                            ignore_errors=true, 
                            all_varchar=true,
                            auto_detect=true)
                    ) TO ? (FORMAT PARQUET, COMPRESSION 'SNAPPY');
                """
                con.execute(query, [filepath, pq_path])
                print(f"    -> Successfully created Parquet: {os.path.basename(pq_path)}")
            except Exception as err:
                print(f"    -> [Warning] Could not parse {file_name}: {err}")
    finally:
        con.close()

    # 6. Upload output directory to Hugging Face
    output_files = glob.glob(os.path.join(output_dir, "*"))
    print(f"\nUploading {len(output_files)} parsed file(s) to Hugging Face...")

    if output_files:
        # Sanitize filename for HF folder path
        clean_name = os.path.splitext(target_filename)[0]
        api.upload_folder(
            folder_path=output_dir,
            repo_id=DEST_REPO,
            repo_type="dataset",
            path_in_repo=f"file_{clean_name}",
            multi_commits=True,
        )
        print("Upload finished successfully!")
    else:
        print("No output files generated to upload.")

finally:
    print("Cleaning up temporary directories...")
    shutil.rmtree(part_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
