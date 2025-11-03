"""
Data Download and Decompression Utility.

This script decompresses gzipped JSONL files for local data processing.
Note: Update the file paths to match your local directory structure.
"""

import gzip
import shutil

# Input and output file paths
# Note: Update these paths according to your local directory structure
input_path = r"C:\Users\Anshul Shinde\Desktop\SEM 7\MMA LAB\multi_modal_review_analyzer\data\raw\meta_Sports_and_Outdoors.jsonl.gz"
output_path = r"C:\Users\Anshul Shinde\Desktop\SEM 7\MMA LAB\multi_modal_review_analyzer\data\raw\meta_Sports_and_Outdoors.jsonl"

# Decompress gzipped file
# Opens the gzipped input file in binary read mode and writes decompressed content to output
with gzip.open(input_path, "rb") as f_in:
    with open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
