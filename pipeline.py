import os
import glob
import subprocess
import duckdb
import shutil
from huggingface_hub import hf_hub_download, HfApi, login

HF_TOKEN = os.getenv("HF_TOKEN")
DEST_REPO = os.getenv("DEST_REPO")
START_PART = int(os.getenv("START_PART", "1"))
END_PART = int(os.getenv("END_PART", "2"))

SOURCE_REPO = "darrifylive/Father-of-All-Breache-FOAB"

login(token=HF_TOKEN)
api = HfApi()

api.create_repo(repo_id=DEST_REPO, repo_type="dataset", private=True, exist_ok=True)

print(f"=== Starting Processing Pipeline for Parts {START_PART} through {END_PART} ===")

# Always ensure part 001 is available as the header index
part1_filename = "xpolite-emaildb.part-2-.part001.rar"

for part in range(START_PART, END_PART + 1):
    print(f"\n--------------------------------------------------")
    print(f"--> Processing Part {part} of {END_PART}...")
    
    part_dir = f"./temp_rar_part_{part}"
    extract_dir = f"./temp_extract_{part}"
    output_dir = f"./temp_out_{part}"
    
    os.makedirs(part_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    target_filename = f"xpolite-emaildb.part-2-.part{part:03d}.rar"
    
    try:
        # A. Always download part001.rar (Header)
        print(f"Downloading Archive Header: {part1_filename}")
        hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=part1_filename,
            repo_type="dataset",
            local_dir=part_dir
        )
        
        # Download target part if different from part001
        if part != 1:
            print(f"Downloading Target Part: {target_filename}")
            hf_hub_download(
                repo_id=SOURCE_REPO,
                filename=target_filename,
                repo_type="dataset",
                local_dir=part_dir
            )
        
        # B. Unrar starting from part001
        part1_path = os.path.join(part_dir, part1_filename)
        print("Extracting volume using 7zip...")
        extract_proc = subprocess.run(
            ["7z", "x", part1_path, f"-o{extract_dir}", "-y"],
            capture_output=True,
            text=True
        )
        print(extract_proc.stdout[:500]) # Log extraction output
        
        # C. Convert extracted files to Parquet
        print("Converting extracted files to compressed Parquet format...")
        extracted_files = [
            f for f in glob.glob(f"{extract_dir}/**/*", recursive=True)
            if os.path.isfile(f)
        ]
        
        print(f"Found {len(extracted_files)} extracted files to convert.")
        
        for filepath in extracted_files:
            base_name = os.path.basename(filepath)
            pq_name = f"{output_dir}/{base_name}.parquet"
            
            try:
                con = duckdb.connect()
                con.execute(f"""
                    COPY (SELECT * FROM read_csv_auto('{filepath}', ignore_errors=true, all_varchar=true))
                    TO '{pq_name}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');
                """)
                con.close()
                print(f"  [Converted]: {base_name} -> {base_name}.parquet")
            except Exception as err:
                print(f"  [Warning]: Skipped {base_name}: {err}")

        # D. Build DuckDB Index
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

        # E. Upload to Hugging Face
        upload_files = glob.glob(f"{output_dir}/*")
        if upload_files:
            print(f"Uploading {len(upload_files)} files to Hugging Face...")
            api.upload_folder(
                folder_path=output_dir,
                repo_id=DEST_REPO,
                repo_type="dataset",
                path_in_repo=f"data_batch_{part}",
                multi_commits=True
            )
            print(f"Successfully processed, indexed, and uploaded Part {part}!")
        else:
            print(f"No files generated in {output_dir} to upload.")

    except Exception as e:
        print(f"Error encountered while processing Part {part}: {e}")

    finally:
        print("Cleaning up local disk space...")
        shutil.rmtree(part_dir, ignore_errors=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)

print("\n=== Workflow Batch Execution Complete! ===")
