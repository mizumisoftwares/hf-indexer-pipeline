import os
import glob
import subprocess
import duckdb
import shutil
from huggingface_hub import hf_hub_download, HfApi, login

HF_TOKEN="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"

HF_TOKEN = os.getenv("HF_TOKEN")
DEST_REPO = os.getenv("DEST_REPO", "sarveshmgkvp/extracted-foab-parquet")
START_PART = int(os.getenv("START_PART", "1"))
END_PART = int(os.getenv("END_PART", "1"))

# Updated to your actual bucket dataset on Hugging Face
SOURCE_REPO = "sarveshmgkvp/Father-of-All-Breache-FOAB-bucket"

login(token=HF_TOKEN)
api = HfApi()

api.create_repo(repo_id=DEST_REPO, repo_type="dataset", private=True, exist_ok=True)

print(f"=== Starting Processing Pipeline for Parts {START_PART} through {END_PART} ===")

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
        # A. Download Header Part 1
        print(f"Downloading Archive Header from {SOURCE_REPO}: {part1_filename}")
        hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=part1_filename,
            repo_type="dataset",
            local_dir=part_dir
        )
        
        # Download target part if different
        if part != 1:
            print(f"Downloading Target Part: {target_filename}")
            hf_hub_download(
                repo_id=SOURCE_REPO,
                filename=target_filename,
                repo_type="dataset",
                local_dir=part_dir
            )
        
        # B. Extract starting from Part 1
        part1_path = os.path.join(part_dir, part1_filename)
        print("Extracting volume using 7zip...")
        subprocess.run(
            ["7z", "x", part1_path, f"-o{extract_dir}", "-y"],
            check=False
        )
        
        # C. Recursively collect ALL extracted files across all subfolders
        extracted_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                extracted_files.append(os.path.join(root, file))
        
        print(f"Found {len(extracted_files)} extracted file(s).")
        
        # D. Convert each file to Parquet & aggregate into DuckDB
        con = duckdb.connect(f"{output_dir}/index_part_{part}.duckdb")
        table_created = False
        
        for idx, filepath in enumerate(extracted_files):
            file_name = os.path.basename(filepath)
            
            # Skip hidden files or previous output files
            if file_name.startswith('.') or file_name.endswith(('.duckdb', '.parquet')):
                continue
                
            pq_name = f"{output_dir}/file_{idx}_{file_name}.parquet"
            print(f"  [Processing]: {file_name}")
            
            try:
                # 1. Convert to Parquet
                temp_con = duckdb.connect()
                temp_con.execute(f"""
                    COPY (SELECT * FROM read_csv_auto('{filepath}', ignore_errors=true, all_varchar=true))
                    TO '{pq_name}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');
                """)
                temp_con.close()
                print(f"    -> Generated Parquet: {os.path.basename(pq_name)}")
                
                # 2. Append to DuckDB index table
                if not table_created:
                    con.execute(f"""
                        CREATE TABLE email_records AS 
                        SELECT * FROM read_csv_auto('{filepath}', ignore_errors=true, all_varchar=true);
                    """)
                    table_created = True
                else:
                    con.execute(f"""
                        INSERT INTO email_records 
                        SELECT * FROM read_csv_auto('{filepath}', ignore_errors=true, all_varchar=true);
                    """)
            except Exception as err:
                print(f"    -> [Warning] Failed to parse {file_name}: {err}")

        # Build Index if data exists
        if table_created:
            try:
                con.execute("CREATE INDEX idx_col0 ON email_records(column0);")
                print("  [Indexed]: Index created on primary column.")
            except Exception as idx_err:
                print(f"  [Index Notice]: {idx_err}")
        
        con.close()

        # E. Upload outputs to Hugging Face
        output_files = glob.glob(f"{output_dir}/*")
        print(f"Total files ready for upload in batch {part}: {len(output_files)}")
        
        if output_files:
            api.upload_folder(
                folder_path=output_dir,
                repo_id=DEST_REPO,
                repo_type="dataset",
                path_in_repo=f"data_batch_{part}",
                multi_commits=True
            )
            print(f"Successfully uploaded batch {part} to Hugging Face!")
        else:
            print("No valid output files generated to upload.")

    except Exception as e:
        print(f"Error processing Part {part}: {e}")

    finally:
        print("Cleaning up local disk space...")
        shutil.rmtree(part_dir, ignore_errors=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)

print("\n=== Pipeline Execution Finished ===")
