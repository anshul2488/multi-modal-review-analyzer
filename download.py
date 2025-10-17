import gzip
import shutil

input_path = r"C:\Users\Anshul Shinde\Desktop\SEM 7\MMA LAB\multi_modal_review_analyzer\data\raw\meta_Sports_and_Outdoors.jsonl.gz"
output_path = r"C:\Users\Anshul Shinde\Desktop\SEM 7\MMA LAB\multi_modal_review_analyzer\data\raw\meta_Sports_and_Outdoors.jsonl"

with gzip.open(input_path, "rb") as f_in:
    with open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
