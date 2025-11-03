"""
Advanced Fusion Model Architectures for Multimodal Review Analysis.

This module implements multiple fusion strategies for combining text and numerical
features in multimodal learning, including Early Fusion, Late Fusion, Hybrid Fusion,
Cross-Modal Transformer, and Ensemble models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusionModel(nn.Module):
    """
    Early Fusion Model: Concatenates text and numerical features at input level.
    
    This architecture combines modalities before deep processing, making it suitable
    when modalities are highly correlated. Features include layer normalization,
    sequential feature transformation, and residual connections.
    
    Args:
        text_dim (int): Dimension of text embeddings
        num_dim (int): Dimension of numerical features
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension (number of classes or regression output)
        dropout (float): Dropout probability for regularization (default: 0.3)
    """
    def __init__(self, text_dim, num_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        # Feature normalization
        self.text_norm = nn.LayerNorm(text_dim)
        self.num_norm = nn.LayerNorm(num_dim)
        
        # Feature transformation
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.num_fc = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion layer with residual connection
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text_emb, num_feat):
        """
        Forward pass through early fusion model.
        
        Args:
            text_emb (torch.Tensor): Text embeddings of shape (batch_size, text_dim)
            num_feat (torch.Tensor): Numerical features of shape (batch_size, num_dim)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_dim)
        """
        # Normalize inputs for stable training
        t_norm = self.text_norm(text_emb)
        n_norm = self.num_norm(num_feat)
        
        # Transform features through fully connected layers
        t = self.text_fc(t_norm)
        n = self.num_fc(n_norm)
        
        # Early fusion: concatenate modalities before processing
        x = torch.cat([t, n], dim=1)
        x = self.fusion(x)
        
        # Final output projection
        return self.output(x)

class LateFusionModel(nn.Module):
    """
    Late Fusion Model: Processes text and numerical features separately, then combines at decision level.
    
    This architecture allows each modality to be processed independently before fusion,
    making it suitable when modalities have different optimal processing strategies.
    Features include separate processing branches and attention-based fusion mechanism.
    
    Args:
        text_dim (int): Dimension of text embeddings
        num_dim (int): Dimension of numerical features
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension
        dropout (float): Dropout probability for regularization (default: 0.3)
    """
    def __init__(self, text_dim, num_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        
        # Separate processing branches
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.num_branch = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Decision-level fusion with attention
        self.fusion_attention = nn.Sequential(
            nn.Linear(2 * output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # attention weights for text and num
            nn.Softmax(dim=1)
        )
        
        self.final_output = nn.Linear(output_dim, output_dim)
        
    def forward(self, text_emb, num_feat):
        """
        Forward pass through late fusion model.
        
        Args:
            text_emb (torch.Tensor): Text embeddings of shape (batch_size, text_dim)
            num_feat (torch.Tensor): Numerical features of shape (batch_size, num_dim)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_dim)
        """
        # Process each modality separately through independent branches
        text_out = self.text_branch(text_emb)
        num_out = self.num_branch(num_feat)
        
        # Compute attention weights for modality fusion
        # Attention determines the relative importance of each modality
        combined = torch.cat([text_out, num_out], dim=1)
        attention_weights = self.fusion_attention(combined)
        
        # Apply attention weights to each modality's output
        weighted_text = attention_weights[:, 0:1] * text_out
        weighted_num = attention_weights[:, 1:2] * num_out
        
        # Combine weighted modalities
        fused = weighted_text + weighted_num
        output = self.final_output(fused)
        
        return output

class HybridFusionModel(nn.Module):
    """
    Hybrid Fusion Model: Combines early and late fusion with cross-modal attention.
    
    This architecture balances benefits of both early and late fusion by using
    bidirectional cross-modal attention, self-attention, and gating mechanisms
    to control information flow between modalities.
    
    Args:
        text_dim (int): Dimension of text embeddings
        num_dim (int): Dimension of numerical features
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout probability for regularization (default: 0.3)
    """
    def __init__(self, text_dim, num_dim, hidden_dim, output_dim, num_heads=8, dropout=0.3):
        super().__init__()
        
        # Input processing
        self.text_norm = nn.LayerNorm(text_dim)
        self.num_norm = nn.LayerNorm(num_dim)
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.num_proj = nn.Linear(num_dim, hidden_dim)
        
        # Cross-modal attention layers
        self.text_to_num_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.num_to_text_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Self-attention for each modality
        self.text_self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.num_self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text_emb, num_feat):
        """
        Forward pass through hybrid fusion model.
        
        Args:
            text_emb (torch.Tensor): Text embeddings of shape (batch_size, text_dim)
            num_feat (torch.Tensor): Numerical features of shape (batch_size, num_dim)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_dim)
        """
        # Normalize and project inputs to common hidden dimension
        t_norm = self.text_norm(text_emb)
        n_norm = self.num_norm(num_feat)
        
        # Project to hidden dimension and add sequence dimension for attention
        # Shape: (batch_size, 1, hidden_dim)
        t_proj = self.text_proj(t_norm).unsqueeze(1)
        n_proj = self.num_proj(n_norm).unsqueeze(1)
        
        # Self-attention within each modality to capture intra-modal dependencies
        t_self, _ = self.text_self_attention(t_proj, t_proj, t_proj)
        n_self, _ = self.num_self_attention(n_proj, n_proj, n_proj)
        
        # Cross-modal attention: each modality attends to the other
        t_attended_by_num, _ = self.text_to_num_attention(query=t_self, key=n_self, value=n_self)
        n_attended_by_text, _ = self.num_to_text_attention(query=n_self, key=t_self, value=t_self)
        
        # Combine original and attended features with residual connections
        t_combined = t_self + t_attended_by_num
        n_combined = n_self + n_attended_by_text
        
        # Remove sequence dimension to get (batch_size, hidden_dim)
        t_combined = t_combined.squeeze(1)
        n_combined = n_combined.squeeze(1)
        
        # Gating mechanism: learn how much to weight each modality
        gate_input = torch.cat([t_combined, n_combined], dim=1)
        gate_weights = self.gate(gate_input)  # Values between 0 and 1
        
        # Apply learned gates to control information flow
        t_gated = gate_weights * t_combined
        n_gated = (1 - gate_weights) * n_combined
        
        # Final fusion: concatenate and process through fusion layers
        fused = torch.cat([t_gated, n_gated], dim=1)
        fused = self.fusion_layers(fused)
        
        return self.output(fused)

class CrossModalTransformer(nn.Module):
    """
    Cross-Modal Transformer: Advanced transformer-based cross-modal fusion architecture.
    
    Uses transformer architecture with learned positional encoding to model complex
    cross-modal interactions and long-range dependencies. Processes text and numerical
    features as a sequence with multi-layer transformer blocks.
    
    Args:
        text_dim (int): Dimension of text embeddings
        num_dim (int): Dimension of numerical features
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension
        num_layers (int): Number of transformer layers (default: 3)
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout probability for regularization (default: 0.1)
    """
    def __init__(self, text_dim, num_dim, hidden_dim, output_dim, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Input embeddings
        self.text_embedding = nn.Linear(text_dim, hidden_dim)
        self.num_embedding = nn.Linear(num_dim, hidden_dim)
        
        # Positional encoding for sequence modeling
        self.pos_encoding = nn.Parameter(torch.randn(2, hidden_dim))
        
        # Cross-modal transformer layers
        self.transformer_layers = nn.ModuleList([
            CrossModalTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text_emb, num_feat):
        """
        Forward pass through cross-modal transformer.
        
        Args:
            text_emb (torch.Tensor): Text embeddings of shape (batch_size, text_dim)
            num_feat (torch.Tensor): Numerical features of shape (batch_size, num_dim)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_dim)
        """
        # Embed inputs to common hidden dimension
        t_emb = self.text_embedding(text_emb)  # Shape: (batch_size, hidden_dim)
        n_emb = self.num_embedding(num_feat)   # Shape: (batch_size, hidden_dim)
        
        # Add learned positional encoding to distinguish text and numerical modalities
        t_pos = t_emb + self.pos_encoding[0:1]  # Text position encoding
        n_pos = n_emb + self.pos_encoding[1:2]  # Numerical position encoding
        
        # Stack as sequence: treat modalities as sequence elements
        # Shape: (batch_size, 2, hidden_dim)
        sequence = torch.stack([t_pos, n_pos], dim=1)
        
        # Apply transformer layers for cross-modal interaction
        for layer in self.transformer_layers:
            sequence = layer(sequence)
        
        # Global average pooling over sequence dimension
        # Aggregates information from both modalities
        output = sequence.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        
        # Final output projection
        return self.output_proj(output)

class CrossModalTransformerLayer(nn.Module):
    """
    Transformer layer for cross-modal processing.
    
    Implements a standard transformer encoder layer with self-attention and
    feed-forward network, including residual connections and layer normalization.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        # Multi-head self-attention mechanism
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        # Layer normalization for attention and feed-forward
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network with expansion factor of 4
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Forward pass through transformer layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Tensor of same shape as input after processing
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class EnsembleFusionModel(nn.Module):
    """
    Ensemble Fusion Model: Combines multiple fusion strategies with learnable weights.
    
    Creates an ensemble of all fusion models (Early, Late, Hybrid, Cross-Modal Transformer)
    and learns optimal weights for combining their predictions. Uses temperature scaling
    for soft weight distribution.
    
    Args:
        text_dim (int): Dimension of text embeddings
        num_dim (int): Dimension of numerical features
        hidden_dim (int): Hidden layer dimension (default: 128)
        output_dim (int): Output dimension (default: 1)
        dropout (float): Dropout probability for regularization (default: 0.2)
    """
    
    def __init__(self, text_dim, num_dim, hidden_dim=128, output_dim=1, dropout=0.2):
        super().__init__()
        
        # Individual fusion models
        self.early_fusion = EarlyFusionModel(text_dim, num_dim, hidden_dim, output_dim, dropout)
        self.late_fusion = LateFusionModel(text_dim, num_dim, hidden_dim, output_dim, dropout)
        self.hybrid_fusion = HybridFusionModel(text_dim, num_dim, hidden_dim, output_dim, dropout)
        self.cross_modal_transformer = CrossModalTransformer(text_dim, num_dim, hidden_dim, output_dim, dropout)
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(output_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Temperature scaling for ensemble weights
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, text_features, numerical_features):
        """
        Forward pass through ensemble fusion model.
        
        Args:
            text_features (torch.Tensor): Text embeddings
            numerical_features (torch.Tensor): Numerical features
            
        Returns:
            torch.Tensor: Ensemble prediction combining all fusion strategies
        """
        # Get predictions from each individual fusion model
        early_pred = self.early_fusion(text_features, numerical_features)
        late_pred = self.late_fusion(text_features, numerical_features)
        hybrid_pred = self.hybrid_fusion(text_features, numerical_features)
        cross_modal_pred = self.cross_modal_transformer(text_features, numerical_features)
        
        # Apply temperature scaling to ensemble weights for smooth distribution
        # Temperature > 1 makes distribution more uniform, < 1 makes it more peaked
        weights = F.softmax(self.ensemble_weights / self.temperature, dim=0)
        
        # Weighted combination of predictions based on learned ensemble weights
        weighted_pred = (weights[0] * early_pred + 
                        weights[1] * late_pred + 
                        weights[2] * hybrid_pred + 
                        weights[3] * cross_modal_pred)
        
        # Concatenate all predictions for learned final fusion
        all_predictions = torch.cat([early_pred, late_pred, hybrid_pred, cross_modal_pred], dim=1)
        final_pred = self.final_fusion(all_predictions)
        
        # Combine weighted average (70%) and learned fusion (30%)
        ensemble_pred = 0.7 * weighted_pred + 0.3 * final_pred
        
        return ensemble_pred

class AdaptiveEnsembleModel(nn.Module):
    """
    Adaptive Ensemble Model: Learns when to use which fusion strategy dynamically.
    
    Uses a gating network to adaptively select the best fusion strategy based on
    input characteristics. Also provides confidence estimation for prediction reliability.
    
    Args:
        text_dim (int): Dimension of text embeddings
        num_dim (int): Dimension of numerical features
        hidden_dim (int): Hidden layer dimension (default: 128)
        output_dim (int): Output dimension (default: 1)
        dropout (float): Dropout probability for regularization (default: 0.2)
    
    Returns:
        tuple: (prediction, confidence, gate_weights) - Prediction with confidence and gating weights
    """
    
    def __init__(self, text_dim, num_dim, hidden_dim=128, output_dim=1, dropout=0.2):
        super().__init__()
        
        # Individual fusion models
        self.early_fusion = EarlyFusionModel(text_dim, num_dim, hidden_dim, output_dim, dropout)
        self.late_fusion = LateFusionModel(text_dim, num_dim, hidden_dim, output_dim, dropout)
        self.hybrid_fusion = HybridFusionModel(text_dim, num_dim, hidden_dim, output_dim, dropout)
        self.cross_modal_transformer = CrossModalTransformer(text_dim, num_dim, hidden_dim, output_dim, dropout)
        
        # Gating network to decide which model to use
        self.gating_network = nn.Sequential(
            nn.Linear(text_dim + num_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),  # 4 fusion strategies
            nn.Softmax(dim=1)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(text_dim + num_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_features, numerical_features):
        """
        Forward pass through adaptive ensemble model.
        
        Args:
            text_features (torch.Tensor): Text embeddings
            numerical_features (torch.Tensor): Numerical features
            
        Returns:
            tuple: (prediction, confidence, gate_weights)
                - prediction: Final ensemble prediction
                - confidence: Confidence score (0 to 1)
                - gate_weights: Learned gating weights for each fusion strategy
        """
        # Combine input features for gating network input
        combined_input = torch.cat([text_features, numerical_features], dim=1)
        
        # Get gating weights: dynamically determines which fusion strategy to emphasize
        # Shape: (batch_size, 4) - weights for each of 4 fusion strategies
        gate_weights = self.gating_network(combined_input)
        
        # Get predictions from each individual fusion model
        early_pred = self.early_fusion(text_features, numerical_features)
        late_pred = self.late_fusion(text_features, numerical_features)
        hybrid_pred = self.hybrid_fusion(text_features, numerical_features)
        cross_modal_pred = self.cross_modal_transformer(text_features, numerical_features)
        
        # Stack all predictions: shape (batch_size, 4, output_dim)
        all_predictions = torch.stack([early_pred.squeeze(), late_pred.squeeze(), 
                                     hybrid_pred.squeeze(), cross_modal_pred.squeeze()], dim=1)
        
        # Apply gating weights: weighted sum based on learned strategy selection
        gated_pred = torch.sum(all_predictions * gate_weights, dim=1, keepdim=True)
        
        # Estimate prediction confidence based on input characteristics
        confidence = self.confidence_estimator(combined_input)
        
        # Return prediction with confidence and gating weights for interpretability
        return gated_pred, confidence, gate_weights
