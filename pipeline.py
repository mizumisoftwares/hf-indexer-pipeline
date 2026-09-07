import os
import glob
import subprocess
import duckdb
import shutil
from huggingface_hub import hf_hub_download, HfApi, login

# Read runtime secrets and workflow parameters
HF_TOKEN = os.getenv("HF_TOKEN")
DEST_REPO = os.getenv("DEST_REPO")
START_PART = int(os.getenv("START_PART", "1"))
END_PART = int(os.getenv("END_PART", "2"))

SOURCE_REPO = "darrifylive/Father-of-All-Breache-FOAB"

# 1. Authenticate with Hugging Face
login(token=HF_TOKEN)
api = HfApi()

# Ensure destination private dataset repository exists
api.create_repo(repo_id=DEST_REPO, repo_type="dataset", private=True, exist_ok=True)

print(f"=== Starting Processing Pipeline for Parts {START_PART} through {END_PART} ===")

# Process in micro-chunks to ensure disk space is never exceeded
for part in range(START_PART, END_PART + 1):
    print(f"\n--------------------------------------------------")
    print(f"--> Processing Part {part} of {END_PART}...")
    
    # Define ephemeral directories per part
    part_dir = f"./temp_rar_part_{part}"
    extract_dir = f"./temp_extract_{part}"
    output_dir = f"./temp_out_{part}"
    
    os.makedirs(part_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"xpolite-emaildb.part-2-.part{part:03d}.rar"
    
    try:
        # A. Download 1 RAR volume from Hugging Face
        print(f"Downloading: {filename}")
        hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=part_dir
        )
        
        # B. Unrar volume using 7zip
        rar_files = glob.glob(f"{part_dir}/*.rar")
        if rar_files:
            print("Extracting volume...")
            subprocess.run(["7z", "x", rar_files[0], f"-o{extract_dir}", "-y"], check=False)
        
        # C. Convert all extracted text/CSV files into Parquet
        print("Converting raw files to compressed Parquet format...")
        extracted_files = glob.glob(f"{extract_dir}/**/*", recursive=True)
        
        for filepath in extracted_files:
            if os.path.isfile(filepath) and not filepath.endswith(('.parquet', '.duckdb')):
                base_name = os.path.basename(filepath)
                pq_name = f"{output_dir}/{base_name}.parquet"
                
                try:
                    con = duckdb.connect()
                    # Low-memory streaming conversion to compressed Parquet
                    con.execute(f"""
                        COPY (SELECT * FROM read_csv_auto('{filepath}', ignore_errors=true, all_varchar=true))
                        TO '{pq_name}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');
                    """)
                    con.close()
                    print(f"  [Converted]: {base_name} -> {base_name}.parquet")
                except Exception as err:
                    print(f"  [Warning]: Skipped {base_name} due to format issue: {err}")

        # D. Build a DuckDB Database Index for sub-10ms lookup
        print("Building DuckDB ART Search Index...")
        db_path = f"{output_dir}/index_part_{part}.duckdb"
        con = duckdb.connect(db_path)
        
        con.execute(f"""
            CREATE TABLE email_records AS 
            SELECT * FROM read_csv_auto('{extract_dir}/*', ignore_errors=true, all_varchar=true);
        """)
        
        try:
            con.execute("CREATE INDEX idx_col0 ON email_records(column0);")
            print("  [Indexed]: ART Index created on primary search column.")
        except Exception as idx_err:
            print(f"  [Index Notice]: {idx_err}")
        con.close()

        # E. Upload Parquet & Index files directly to Hugging Face
        print("Uploading processed files to Hugging Face...")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=DEST_REPO,
            repo_type="dataset",
            path_in_repo=f"data_batch_{part}",
            multi_commits=True
        )
        print(f"Successfully processed, indexed, and uploaded Part {part}!")

    except Exception as e:
        print(f"Error encountered while processing Part {part}: {e}")

    finally:
        # CRITICAL: Delete local files to free disk space for the next part
        print("Cleaning up local disk space...")
        shutil.rmtree(part_dir, ignore_errors=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)

print("\n=== Workflow Batch Execution Complete! ===")
