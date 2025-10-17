import os

# Base project directory (adjust if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

REVIEWS_FILE = os.path.join(RAW_DATA_DIR, "Sports_and_Outdoors.jsonl")  
METADATA_FILE = os.path.join(RAW_DATA_DIR, "meta_Sports_and_Outdoors.jsonl")

# Model save/load paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
SAVED_MODEL_PATH = os.path.join(MODEL_DIR, "best_fusion_model.pth")

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Device config
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NLP Model
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 128
