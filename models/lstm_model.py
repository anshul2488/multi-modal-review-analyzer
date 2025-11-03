"""
LSTM Model Implementation for Sequential Text Processing.

This module implements a Long Short-Term Memory (LSTM) neural network
for processing sequential text data in the multimodal review analyzer.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM model for sequential text processing.
    
    This model uses LSTM layers to process sequential embeddings and
    extracts the final hidden state for classification or regression tasks.
    
    Args:
        input_size (int): Dimension of input features at each timestep
        hidden_size (int): Number of hidden units in LSTM layers
        num_layers (int): Number of stacked LSTM layers
        output_size (int): Dimension of output (e.g., number of classes or regression output)
        dropout (float): Dropout probability for regularization (default: 0.3)
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        """
        Initialize LSTM model architecture.
        
        Args:
            input_size (int): Dimension of input features at each timestep
            hidden_size (int): Number of hidden units in LSTM layers
            num_layers (int): Number of stacked LSTM layers
            output_size (int): Dimension of output
            dropout (float): Dropout probability for regularization
        """
        super(LSTMModel, self).__init__()
        
        # LSTM layer for sequential processing
        # batch_first=True: input/output format is (batch, seq_len, features)
        # dropout applies to all LSTM layers except the last
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected layer for final output projection
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Process input through LSTM layers
        # Output shape: (batch_size, seq_len, hidden_size)
        # Hidden states are ignored (using _) since we only need the output
        out, _ = self.lstm(x)
        
        # Extract the last timestep output (final hidden state representation)
        # Shape: (batch_size, hidden_size)
        last_timestep = out[:, -1, :]
        
        # Apply fully connected layer for final prediction
        # Shape: (batch_size, output_size)
        output = self.fc(last_timestep)
        
        return output
