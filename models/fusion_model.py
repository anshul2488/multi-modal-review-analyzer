import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyFusionModel(nn.Module):
    """
    Early Fusion: Concatenates text and numerical features at input level
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
        # Normalize inputs
        t_norm = self.text_norm(text_emb)
        n_norm = self.num_norm(num_feat)
        
        # Transform features
        t = self.text_fc(t_norm)
        n = self.num_fc(n_norm)
        
        # Early fusion: concatenate and process
        x = torch.cat([t, n], dim=1)
        x = self.fusion(x)
        
        return self.output(x)

class LateFusionModel(nn.Module):
    """
    Late Fusion: Processes text and numerical features separately, then combines at decision level
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
        # Process each modality separately
        text_out = self.text_branch(text_emb)
        num_out = self.num_branch(num_feat)
        
        # Compute attention weights
        combined = torch.cat([text_out, num_out], dim=1)
        attention_weights = self.fusion_attention(combined)
        
        # Weighted fusion
        weighted_text = attention_weights[:, 0:1] * text_out
        weighted_num = attention_weights[:, 1:2] * num_out
        
        # Final decision
        fused = weighted_text + weighted_num
        output = self.final_output(fused)
        
        return output

class HybridFusionModel(nn.Module):
    """
    Hybrid Fusion: Combines early and late fusion with cross-modal attention
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
        # Normalize and project inputs
        t_norm = self.text_norm(text_emb)
        n_norm = self.num_norm(num_feat)
        
        t_proj = self.text_proj(t_norm).unsqueeze(1)  # [B, 1, H]
        n_proj = self.num_proj(n_norm).unsqueeze(1)   # [B, 1, H]
        
        # Self-attention within each modality
        t_self, _ = self.text_self_attention(t_proj, t_proj, t_proj)
        n_self, _ = self.num_self_attention(n_proj, n_proj, n_proj)
        
        # Cross-modal attention
        t_attended_by_num, _ = self.text_to_num_attention(query=t_self, key=n_self, value=n_self)
        n_attended_by_text, _ = self.num_to_text_attention(query=n_self, key=t_self, value=t_self)
        
        # Combine original and attended features
        t_combined = t_self + t_attended_by_num
        n_combined = n_self + n_attended_by_text
        
        # Squeeze back to [B, H]
        t_combined = t_combined.squeeze(1)
        n_combined = n_combined.squeeze(1)
        
        # Gating mechanism
        gate_input = torch.cat([t_combined, n_combined], dim=1)
        gate_weights = self.gate(gate_input)
        
        # Apply gates
        t_gated = gate_weights * t_combined
        n_gated = (1 - gate_weights) * n_combined
        
        # Final fusion
        fused = torch.cat([t_gated, n_gated], dim=1)
        fused = self.fusion_layers(fused)
        
        return self.output(fused)

class CrossModalTransformer(nn.Module):
    """
    Advanced cross-modal transformer with positional encoding and multiple fusion strategies
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
        # Embed inputs
        t_emb = self.text_embedding(text_emb)  # [B, H]
        n_emb = self.num_embedding(num_feat)   # [B, H]
        
        # Add positional encoding and create sequence
        t_pos = t_emb + self.pos_encoding[0:1]  # [B, H]
        n_pos = n_emb + self.pos_encoding[1:2]  # [B, H]
        
        # Stack as sequence [B, 2, H]
        sequence = torch.stack([t_pos, n_pos], dim=1)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            sequence = layer(sequence)
        
        # Global average pooling
        output = sequence.mean(dim=1)  # [B, H]
        
        return self.output_proj(output)

class CrossModalTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class EnsembleFusionModel(nn.Module):
    """Ensemble model combining multiple fusion strategies"""
    
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
        # Get predictions from each fusion model
        early_pred = self.early_fusion(text_features, numerical_features)
        late_pred = self.late_fusion(text_features, numerical_features)
        hybrid_pred = self.hybrid_fusion(text_features, numerical_features)
        cross_modal_pred = self.cross_modal_transformer(text_features, numerical_features)
        
        # Apply temperature scaling to ensemble weights
        weights = F.softmax(self.ensemble_weights / self.temperature, dim=0)
        
        # Weighted combination of predictions
        weighted_pred = (weights[0] * early_pred + 
                        weights[1] * late_pred + 
                        weights[2] * hybrid_pred + 
                        weights[3] * cross_modal_pred)
        
        # Concatenate all predictions for final fusion
        all_predictions = torch.cat([early_pred, late_pred, hybrid_pred, cross_modal_pred], dim=1)
        final_pred = self.final_fusion(all_predictions)
        
        # Combine weighted and final predictions
        ensemble_pred = 0.7 * weighted_pred + 0.3 * final_pred
        
        return ensemble_pred

class AdaptiveEnsembleModel(nn.Module):
    """Adaptive ensemble that learns when to use which fusion strategy"""
    
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
        # Combine input features for gating
        combined_input = torch.cat([text_features, numerical_features], dim=1)
        
        # Get gating weights
        gate_weights = self.gating_network(combined_input)
        
        # Get predictions from each model
        early_pred = self.early_fusion(text_features, numerical_features)
        late_pred = self.late_fusion(text_features, numerical_features)
        hybrid_pred = self.hybrid_fusion(text_features, numerical_features)
        cross_modal_pred = self.cross_modal_transformer(text_features, numerical_features)
        
        # Stack predictions
        all_predictions = torch.stack([early_pred.squeeze(), late_pred.squeeze(), 
                                     hybrid_pred.squeeze(), cross_modal_pred.squeeze()], dim=1)
        
        # Apply gating weights
        gated_pred = torch.sum(all_predictions * gate_weights, dim=1, keepdim=True)
        
        # Estimate confidence
        confidence = self.confidence_estimator(combined_input)
        
        # Return prediction with confidence
        return gated_pred, confidence, gate_weights
