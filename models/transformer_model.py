"""
Transformer Encoder Model Implementation for Text Processing.

This module implements a Transformer encoder architecture using multi-head
self-attention for processing sequential text embeddings in the multimodal
review analyzer.
"""

import torch
import torch.nn as nn


class TransformerEncoderModel(nn.Module):
    """
    Transformer encoder model for sequential text processing.
    
    This model uses multi-head self-attention mechanism to process sequential
    embeddings, capturing long-range dependencies and contextual information.
    
    Args:
        input_dim (int): Dimension of input features at each timestep
        num_heads (int): Number of attention heads in multi-head attention
        num_layers (int): Number of transformer encoder layers
        hidden_dim (int): Hidden dimension (d_model) for transformer layers
        output_dim (int): Dimension of output (e.g., number of classes or regression output)
        dropout (float): Dropout probability for regularization (default: 0.3)
    """
    
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim, dropout=0.3):
        """
        Initialize Transformer encoder model architecture.
        
        Args:
            input_dim (int): Dimension of input features at each timestep
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            hidden_dim (int): Hidden dimension for transformer
            output_dim (int): Dimension of output
            dropout (float): Dropout probability for regularization
        """
        super().__init__()
        
        # Input projection layer: maps input dimension to transformer hidden dimension
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layer configuration
        # Contains: multi-head self-attention, feed-forward network, layer normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,      # Model dimension
            nhead=num_heads,          # Number of attention heads
            dropout=dropout,          # Dropout rate
            batch_first=True          # Input/output format: (batch, seq_len, features)
        )
        
        # Stack multiple transformer encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection layer: maps transformer hidden dimension to output dimension
        self.output_fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass through the transformer encoder model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Project input to transformer hidden dimension
        # Shape: (batch_size, seq_len, hidden_dim)
        x = self.input_fc(x)
        
        # Process through transformer encoder layers
        # Applies multi-head self-attention and feed-forward networks
        # Shape: (batch_size, seq_len, hidden_dim)
        x = self.transformer(x)
        
        # Global average pooling over sequence dimension
        # Aggregates information from all timesteps
        # Shape: (batch_size, hidden_dim)
        pooled = x.mean(dim=1)
        
        # Apply output projection for final prediction
        # Shape: (batch_size, output_dim)
        output = self.output_fc(pooled)
        
        return output
