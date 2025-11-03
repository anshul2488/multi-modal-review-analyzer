"""
Configuration file for Multimodal Review Analyzer project.

This module contains all configuration parameters including data paths,
model settings, training hyperparameters, and device configuration.
"""

import os
import torch

# ============================================================================
# Base Directory Configuration
# ============================================================================

# Base project directory - automatically detected from this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Data Path Configuration
# ============================================================================

# Root data directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# Raw data directory - contains original JSONL files
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Processed data directory - contains preprocessed Parquet files
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Default data file paths
# Note: Adjust these paths based on your specific dataset
REVIEWS_FILE = os.path.join(RAW_DATA_DIR, "Sports_and_Outdoors.jsonl")  
METADATA_FILE = os.path.join(RAW_DATA_DIR, "meta_Sports_and_Outdoors.jsonl")

# ============================================================================
# Model Path Configuration
# ============================================================================

# Directory for saving trained models
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Path for saving the best fusion model
SAVED_MODEL_PATH = os.path.join(MODEL_DIR, "best_fusion_model.pth")

# ============================================================================
# Training Hyperparameters
# ============================================================================

# Batch size for training (number of samples per batch)
BATCH_SIZE = 64

# Number of training epochs
NUM_EPOCHS = 10

# Learning rate for optimizer
LEARNING_RATE = 0.001

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Device Configuration
# ============================================================================

# Automatically detect and configure compute device (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# NLP Model Configuration
# ============================================================================

# SBERT model name for text embeddings
# Using MiniLM-L6-v2: compact model with good performance
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Maximum sequence length for text processing
MAX_SEQ_LENGTH = 128
