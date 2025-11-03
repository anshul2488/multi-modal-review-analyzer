"""
SBERT Encoder Implementation for Text Embeddings.

This module implements a Sentence-BERT (SBERT) encoder for generating
semantic text embeddings using transformer-based language models.
"""

from transformers import AutoTokenizer, AutoModel
import torch


class SBERTEncoder:
    """
    Sentence-BERT encoder for generating text embeddings.
    
    This class wraps a pre-trained transformer model to encode text into
    fixed-dimensional vector representations suitable for downstream tasks.
    
    Args:
        model_name (str): Name or path of the pre-trained model (default: "sentence-transformers/all-MiniLM-L6-v2")
        device (str): Device to run the model on ("cpu" or "cuda") (default: "cpu")
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        """
        Initialize SBERT encoder with tokenizer and model.
        
        Args:
            model_name (str): HuggingFace model identifier or local path
            device (str): Target device for model execution
        """
        # Initialize tokenizer for text preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load pre-trained transformer model and move to specified device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
        # Store device for later tensor operations
        self.device = device

    def encode(self, texts, max_length=128):
        """
        Encode a list of texts into fixed-dimensional embeddings.
        
        This method tokenizes input texts, processes them through the transformer
        model, and generates sentence embeddings using mean pooling over token
        embeddings weighted by attention masks.
        
        Args:
            texts (list of str): List of input text strings to encode
            max_length (int): Maximum sequence length for tokenization (default: 128)
            
        Returns:
            numpy.ndarray: Array of embeddings with shape (batch_size, embedding_dim)
        """
        # Tokenize input texts with padding and truncation
        # Returns dictionary with 'input_ids' and 'attention_mask'
        inputs = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,           # Pad shorter sequences to max_length
            truncation=True,        # Truncate longer sequences to max_length
            max_length=max_length,  # Maximum sequence length
            return_tensors='pt'      # Return PyTorch tensors
        )
        
        # Move all input tensors to the specified device (GPU/CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Disable gradient computation for inference (faster, less memory)
        with torch.no_grad():
            # Forward pass through the transformer model
            output = self.model(**inputs)
            
            # Extract token embeddings from the last hidden layer
            # Shape: (batch_size, seq_len, hidden_dim)
            emb = output.last_hidden_state
            
            # Expand attention mask to match embedding dimensions
            # Shape: (batch_size, seq_len, hidden_dim)
            mask = inputs['attention_mask'].unsqueeze(-1).expand(emb.size())
            
            # Mean pooling: sum embeddings weighted by attention mask, then divide by mask sum
            # This effectively averages only over non-padding tokens
            # Shape: (batch_size, hidden_dim)
            masked_embeddings = (emb * mask).sum(dim=1) / mask.sum(dim=1)
            
            # Convert to numpy array and return (moved to CPU for compatibility)
            return masked_embeddings.cpu().numpy()
