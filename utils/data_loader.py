import json
import os
from typing import List, Optional

import pandas as pd
import torch

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SBERT_MODEL_NAME, MAX_SEQ_LENGTH, DEVICE
from preprocessing.text_preprocessor import TextPreprocessor
from models.nlp_utils import SBERTEncoder
from preprocessing.feature_engineering import sentiment_features

def load_jsonl(filename):
    """Load a JSONL file into a pandas DataFrame."""
    records = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame.from_records(records)
def jsonl_batch_generator(
    filename,
    batch_size=32768,
    text_column='reviewText',
    num_columns=None,
    label_column=None,
    device=None,
    preprocess_fn=None,
):
    """Yields batches as lists (text), tensors (num/label), moved to device."""
    num_columns = num_columns or []
    texts, nums, labels = [], [], []
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get(text_column, "")
            if preprocess_fn:
                text = preprocess_fn(text)
            num_feats = [float(data.get(col, 0.0)) for col in num_columns]
            label = data.get(label_column, -1) if label_column else -1

            texts.append(text)
            nums.append(num_feats)
            labels.append(label)
            if len(texts) == batch_size:
                nums_tensor = torch.tensor(nums, dtype=torch.float32, device=device)
                labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
                yield texts, nums_tensor, labels_tensor
                texts, nums, labels = [], [], []
        if texts:
            nums_tensor = torch.tensor(nums, dtype=torch.float32, device=device)
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
            yield texts, nums_tensor, labels_tensor


def list_jsonl_files(directory: str = RAW_DATA_DIR) -> List[str]:
    """Return sorted list of .jsonl files in the given directory."""
    try:
        files = [f for f in os.listdir(directory) if f.lower().endswith('.jsonl')]
        files.sort()
        return files
    except FileNotFoundError:
        return []


def _ensure_dir_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _sbert_encode_batch(encoder: SBERTEncoder, texts: List[str], max_length: int) -> pd.DataFrame:
    if not texts:
        return pd.DataFrame()
    emb = encoder.encode(texts, max_length=max_length)  # returns np.ndarray [B, D]
    dim = emb.shape[1]
    columns = [f"emb_{i}" for i in range(dim)]
    return pd.DataFrame(emb, columns=columns)


def process_reviews_to_parquet(
    input_jsonl_path: str,
    output_parquet_path: Optional[str] = None,
    batch_size: int = 2048,
    device: Optional[torch.device] = None,
) -> str:
    """
    Process a reviews JSONL into a Parquet dataset with cleaned text, sentiment,
    and SBERT embeddings. Uses GPU if available.

    Returns the output parquet path.
    """
    if device is None:
        device = DEVICE
    if output_parquet_path is None:
        _ensure_dir_exists(PROCESSED_DATA_DIR)
        base = os.path.splitext(os.path.basename(input_jsonl_path))[0]
        output_parquet_path = os.path.join(PROCESSED_DATA_DIR, f"{base}_processed.parquet")

    preprocessor = TextPreprocessor()
    encoder = SBERTEncoder(model_name=SBERT_MODEL_NAME, device=device)

    cleaned_chunks: List[pd.DataFrame] = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        texts: List[str] = []
        raw_rows: List[dict] = []
        for line in f:
            record = json.loads(line)
            raw_rows.append(record)
            text = record.get('reviewText', '')
            cleaned = preprocessor.clean_text(text)
            cleaned = preprocessor.remove_stopwords(cleaned)
            texts.append(cleaned)

            if len(texts) >= batch_size:
                embeddings_df = _sbert_encode_batch(encoder, texts, MAX_SEQ_LENGTH)
                df = pd.DataFrame(raw_rows)
                df['clean_text'] = texts
                # sentiment
                sentiments = [sentiment_features(t) for t in texts]
                df['sent_polarity'] = [s['polarity'] for s in sentiments]
                df['sent_subjectivity'] = [s['subjectivity'] for s in sentiments]
                # concat embeddings
                df = pd.concat([df.reset_index(drop=True), embeddings_df.reset_index(drop=True)], axis=1)
                cleaned_chunks.append(df)
                texts, raw_rows = [], []

        # tail
        if texts:
            embeddings_df = _sbert_encode_batch(encoder, texts, MAX_SEQ_LENGTH)
            df = pd.DataFrame(raw_rows)
            df['clean_text'] = texts
            sentiments = [sentiment_features(t) for t in texts]
            df['sent_polarity'] = [s['polarity'] for s in sentiments]
            df['sent_subjectivity'] = [s['subjectivity'] for s in sentiments]
            df = pd.concat([df.reset_index(drop=True), embeddings_df.reset_index(drop=True)], axis=1)
            cleaned_chunks.append(df)

    if not cleaned_chunks:
        # If input empty, still create an empty parquet
        empty_df = pd.DataFrame()
        empty_df.to_parquet(output_parquet_path, index=False)
        return output_parquet_path

    final_df = pd.concat(cleaned_chunks, ignore_index=True)
    final_df.to_parquet(output_parquet_path, index=False)
    return output_parquet_path


def ensure_processed_parquet(input_jsonl_path: str) -> str:
    """Create processed parquet if missing; return its path."""
    _ensure_dir_exists(PROCESSED_DATA_DIR)
    base = os.path.splitext(os.path.basename(input_jsonl_path))[0]
    output_parquet_path = os.path.join(PROCESSED_DATA_DIR, f"{base}_processed.parquet")
    if not os.path.exists(output_parquet_path):
        process_reviews_to_parquet(input_jsonl_path=input_jsonl_path, output_parquet_path=output_parquet_path)
    return output_parquet_path


def load_processed_parquet(parquet_path: str) -> pd.DataFrame:
    """Efficiently load a processed parquet file."""
    return pd.read_parquet(parquet_path)
