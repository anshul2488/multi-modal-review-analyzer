import os
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import all necessary modules
from config import (
    REVIEWS_FILE, METADATA_FILE, DEVICE, SBERT_MODEL_NAME, MAX_SEQ_LENGTH, 
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODEL_PATH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
)
from utils.data_loader import (
    load_jsonl, list_jsonl_files, ensure_processed_parquet, load_processed_parquet,
    process_reviews_to_parquet
)
from preprocessing.text_preprocessor import TextPreprocessor
from models.nlp_utils import SBERTEncoder
from models.fusion_model import (
    EarlyFusionModel, LateFusionModel, HybridFusionModel, CrossModalTransformer,
    EnsembleFusionModel, AdaptiveEnsembleModel
)
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerEncoderModel
from preprocessing.feature_engineering import (
    sentiment_features, advanced_sentiment_features, cross_modal_feature_engineering,
    extract_linguistic_features, create_fusion_features
)
from utils.metrics import (
    classification_metrics, regression_metrics, cross_modal_evaluation_metrics,
    advanced_model_comparison, generate_model_report
)
from utils.visualization import (
    plot_rating_distribution, create_model_comparison_chart, 
    create_fusion_technique_comparison, create_performance_radar_chart,
    export_results_to_html
)

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)
os.makedirs('reports', exist_ok=True)

class ModelTrainer:
    """Comprehensive model trainer with improved output and visualization"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.encoder = SBERTEncoder(model_name=SBERT_MODEL_NAME, device=DEVICE)
        self.trained_models = {}
        self.model_metrics = {}
        self.training_history = {}
        
        print(f"🚀 Model Trainer initialized on device: {DEVICE}")
        print(f"📊 SBERT Model: {SBERT_MODEL_NAME}")
        
    def get_embedding_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith('emb_')]
    
    def build_dataset_from_df(self, df: pd.DataFrame, task_type: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build dataset from processed DataFrame with detailed logging"""
        emb_cols = self.get_embedding_columns(df)
        if not emb_cols:
            raise ValueError("No embedding columns found in data")
            
        print(f"📈 Dataset Info:")
        print(f"   - Total samples: {len(df)}")
        print(f"   - Embedding dimensions: {len(emb_cols)}")
        print(f"   - Available columns: {list(df.columns)}")
        print(f"   - Task type: {task_type}")
        
        X_text = torch.tensor(df[emb_cols].values, dtype=torch.float32, device=DEVICE)
        
        # Enhanced numerical features with cross-modal feature engineering
        num_feats = ['sent_polarity', 'sent_subjectivity']
        
        # Find the correct text column name
        text_columns = ['reviewText', 'review_text', 'text', 'review', 'content']
        text_column = None
        for col in text_columns:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            # If no standard text column found, use the first string column
            for col in df.columns:
                if df[col].dtype == 'object' and col not in ['asin', 'reviewerID', 'reviewTime']:
                    text_column = col
                    break
        
        if text_column is None:
            print("⚠️  Warning: No text column found, using dummy text")
            text_column = 'dummy_text'
            df[text_column] = 'sample review text'
        
        print(f"📝 Using text column: {text_column}")
        
        # Add advanced features if not present
        if 'word_count' not in df.columns:
            df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        if 'char_count' not in df.columns:
            df['char_count'] = df[text_column].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        if 'avg_word_length' not in df.columns:
            df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1e-6)
        
        # Ensure all required features exist
        enhanced_num_feats = num_feats + ['word_count', 'char_count', 'avg_word_length']
        for nf in enhanced_num_feats:
            if nf not in df.columns:
                if nf in ['sent_polarity', 'sent_subjectivity']:
                    # Generate basic sentiment features if not available
                    df[nf] = 0.0
                    print(f"⚠️  Warning: {nf} not found, using default value 0.0")
                else:
                    df[nf] = 0.0
        
        X_num = torch.tensor(df[enhanced_num_feats].values, dtype=torch.float32, device=DEVICE)
        
        # Labels with detailed statistics
        if task_type == 'classification':
            if 'overall' in df.columns:
                y = (df['overall'] >= 4).astype(int).values
                positive_ratio = y.mean()
                print(f"   - Positive samples: {y.sum()}/{len(y)} ({positive_ratio:.2%})")
            elif 'rating' in df.columns:
                y = (df['rating'] >= 4).astype(int).values
                positive_ratio = y.mean()
                print(f"   - Positive samples: {y.sum()}/{len(y)} ({positive_ratio:.2%})")
            else:
                y = np.random.randint(0, 2, len(df))
                print(f"   - Using random labels (no 'overall' or 'rating' column found)")
            y = torch.tensor(y, dtype=torch.long, device=DEVICE)
        else:  # regression
            if 'overall' in df.columns:
                y_vals = df['overall'].values
                y = torch.tensor(y_vals, dtype=torch.float32, device=DEVICE)
                print(f"   - Rating range: {y_vals.min():.1f} - {y_vals.max():.1f}")
                print(f"   - Rating mean: {y_vals.mean():.2f} ± {y_vals.std():.2f}")
            elif 'rating' in df.columns:
                y_vals = df['rating'].values
                y = torch.tensor(y_vals, dtype=torch.float32, device=DEVICE)
                print(f"   - Rating range: {y_vals.min():.1f} - {y_vals.max():.1f}")
                print(f"   - Rating mean: {y_vals.mean():.2f} ± {y_vals.std():.2f}")
            else:
                y = torch.randn(len(df), dtype=torch.float32, device=DEVICE) * 2 + 3  # Random ratings 1-5
                print(f"   - Using random ratings (no 'overall' or 'rating' column found)")
                
        return X_text, X_num, y
    
    def prepare_sequence_from_embeddings(self, X_text: torch.Tensor, seq_len: int = 16) -> torch.Tensor:
        """Convert flat embeddings to sequence format for LSTM/Transformer"""
        n, d = X_text.shape
        if d % seq_len != 0:
            new_d = ((d + seq_len - 1) // seq_len) * seq_len
            pad = new_d - d
            X_text = torch.nn.functional.pad(X_text, (0, pad))
            d = new_d
        step = d // seq_len
        return X_text.view(n, seq_len, step)
    
    def create_model(self, model_name: str, text_dim: int, num_dim: int, task_type: str) -> nn.Module:
        """Create model based on name and task with detailed architecture info"""
        hidden_dim = 256
        
        if task_type == 'classification':
            output_dim = 2
        else:
            output_dim = 1
            
        print(f"🏗️  Creating {model_name} model:")
        print(f"   - Input dimensions: text={text_dim}, num={num_dim}")
        print(f"   - Hidden dimension: {hidden_dim}")
        print(f"   - Output dimension: {output_dim}")
            
        if model_name == 'EarlyFusion':
            model = EarlyFusionModel(text_dim, num_dim, hidden_dim, output_dim).to(DEVICE)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   - Total parameters: {total_params:,}")
            return model
            
        elif model_name == 'LateFusion':
            model = LateFusionModel(text_dim, num_dim, hidden_dim, output_dim).to(DEVICE)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   - Total parameters: {total_params:,}")
            return model
            
        elif model_name == 'HybridFusion':
            model = HybridFusionModel(text_dim, num_dim, hidden_dim, output_dim).to(DEVICE)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   - Total parameters: {total_params:,}")
            return model
            
        elif model_name == 'CrossModalTransformer':
            model = CrossModalTransformer(text_dim, num_dim, hidden_dim, output_dim).to(DEVICE)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   - Total parameters: {total_params:,}")
            return model
            
        elif model_name == 'EnsembleFusion':
            model = EnsembleFusionModel(text_dim, num_dim, hidden_dim, output_dim).to(DEVICE)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   - Total parameters: {total_params:,}")
            return model
            
        elif model_name == 'AdaptiveEnsemble':
            model = AdaptiveEnsembleModel(text_dim, num_dim, hidden_dim, output_dim).to(DEVICE)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   - Total parameters: {total_params:,}")
            return model
            
        elif model_name == 'LSTM':
            seq_dim = self.prepare_sequence_from_embeddings(
                torch.zeros(1, text_dim, device=DEVICE)
            ).shape[2]
            
            print(f"   - Sequence dimension: {seq_dim}")
            print(f"   - Sequence length: 16")
            
            base_model = LSTMModel(
                input_size=seq_dim, 
                hidden_size=192, 
                num_layers=2, 
                output_size=hidden_dim, 
                dropout=0.2
            ).to(DEVICE)
            
            head = nn.Linear(hidden_dim + num_dim, output_dim).to(DEVICE)
            
            class LSTMWrapper(nn.Module):
                def __init__(self, base, head):
                    super().__init__()
                    self.base = base
                    self.head = head
                    
                def forward(self, x_text, x_num):
                    x_seq = self.prepare_sequence_from_embeddings(x_text)
                    h = self.base(x_seq)
                    return self.head(torch.cat([h, x_num], dim=1))
                    
            wrapper = LSTMWrapper(base_model, head)
            wrapper.prepare_sequence_from_embeddings = self.prepare_sequence_from_embeddings
            total_params = sum(p.numel() for p in wrapper.parameters())
            print(f"   - Total parameters: {total_params:,}")
            return wrapper
            
        elif model_name == 'Transformer':
            seq_dim = self.prepare_sequence_from_embeddings(
                torch.zeros(1, text_dim, device=DEVICE)
            ).shape[2]
            
            print(f"   - Sequence dimension: {seq_dim}")
            print(f"   - Sequence length: 16")
            print(f"   - Attention heads: 4")
            print(f"   - Transformer layers: 2")
            
            base_model = TransformerEncoderModel(
                input_dim=seq_dim,
                num_heads=4,
                num_layers=2,
                hidden_dim=256,
                output_dim=hidden_dim,
                dropout=0.2
            ).to(DEVICE)
            
            head = nn.Linear(hidden_dim + num_dim, output_dim).to(DEVICE)
            
            class TransformerWrapper(nn.Module):
                def __init__(self, base, head):
                    super().__init__()
                    self.base = base
                    self.head = head
                    
                def forward(self, x_text, x_num):
                    x_seq = self.prepare_sequence_from_embeddings(x_text)
                    h = self.base(x_seq)
                    return self.head(torch.cat([h, x_num], dim=1))
                    
            wrapper = TransformerWrapper(base_model, head)
            wrapper.prepare_sequence_from_embeddings = self.prepare_sequence_from_embeddings
            total_params = sum(p.numel() for p in wrapper.parameters())
            print(f"   - Total parameters: {total_params:,}")
            return wrapper
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_model(self, model_name: str, df: pd.DataFrame, task_type: str, 
                   epochs: int = NUM_EPOCHS, max_rows: int = 10000) -> Dict:
        """Train a single model with detailed progress tracking"""
        print(f"\n🎯 Training {model_name} for {task_type} task...")
        print("=" * 60)
        
        # Subsample for faster training
        if len(df) > max_rows:
            print(f"📊 Subsampling from {len(df)} to {max_rows} rows for faster training")
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
            
        X_text, X_num, y = self.build_dataset_from_df(df, task_type)
        text_dim = X_text.shape[1]
        num_dim = X_num.shape[1]
        
        # Create model
        model = self.create_model(model_name, text_dim, num_dim, task_type)
        
        # Setup training
        if task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
            
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        
        # Train/val split
        n = X_text.shape[0]
        split = int(0.8 * n)
        indices = torch.randperm(n, device=DEVICE)
        train_idx, val_idx = indices[:split], indices[split:]
        
        print(f"📊 Train/Val split: {len(train_idx)}/{len(val_idx)} samples")
        print(f"🔧 Optimizer: Adam (lr={LEARNING_RATE})")
        print(f"📈 Scheduler: StepLR (step=2, gamma=0.9)")
        
        # Training loop with tqdm
        train_losses, val_losses = [], []
        train_metrics, val_metrics = [], []
        
        epoch_pbar = tqdm(range(epochs), desc=f"Training {model_name}", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for epoch in epoch_pbar:
            # Training
            model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []
            
            train_pbar = tqdm(range(0, len(train_idx), BATCH_SIZE), 
                             desc=f"Epoch {epoch+1}/{epochs} - Train", 
                             leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
            
            for i in train_pbar:
                batch_idx = train_idx[i:i+BATCH_SIZE]
                xb_text = X_text[batch_idx]
                xb_num = X_num[batch_idx]
                yb = y[batch_idx]
                
                optimizer.zero_grad()
                outputs = model(xb_text, xb_num)
                
                if task_type == 'classification':
                    loss = criterion(outputs, yb)
                    preds = outputs.argmax(dim=1)
                else:
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, yb)
                    preds = outputs
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(preds.detach().cpu().numpy())
                train_targets.extend(yb.detach().cpu().numpy())
                
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            
            val_pbar = tqdm(range(0, len(val_idx), BATCH_SIZE), 
                           desc=f"Epoch {epoch+1}/{epochs} - Val", 
                           leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
            
            with torch.no_grad():
                for i in val_pbar:
                    batch_idx = val_idx[i:i+BATCH_SIZE]
                    xb_text = X_text[batch_idx]
                    xb_num = X_num[batch_idx]
                    yb = y[batch_idx]
                    
                    outputs = model(xb_text, xb_num)
                    
                    if task_type == 'classification':
                        loss = criterion(outputs, yb)
                        preds = outputs.argmax(dim=1)
                    else:
                        outputs = outputs.squeeze(-1)
                        loss = criterion(outputs, yb)
                        preds = outputs
                    
                    val_loss += loss.item()
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(yb.cpu().numpy())
                    
                    val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Calculate metrics
            if task_type == 'classification':
                train_metric = classification_metrics(train_targets, train_preds)
                val_metric = classification_metrics(val_targets, val_preds)
                metric_name = 'Accuracy'
                metric_val = val_metric.get('Accuracy', 0)
            else:
                train_metric = regression_metrics(train_targets, train_preds)
                val_metric = regression_metrics(val_targets, val_preds)
                metric_name = 'MAE'
                metric_val = val_metric.get('MAE', 0)
            
            train_losses.append(train_loss / len(train_idx) * BATCH_SIZE)
            val_losses.append(val_loss / len(val_idx) * BATCH_SIZE)
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)
            
            scheduler.step()
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_losses[-1]:.4f}',
                'Val_Loss': f'{val_losses[-1]:.4f}',
                f'Val_{metric_name}': f'{metric_val:.4f}'
            })
        
        epoch_pbar.close()
        
        # Save model
        model_path = os.path.join(os.path.dirname(SAVED_MODEL_PATH), f"{model_name}_{task_type}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'task_type': task_type,
            'text_dim': text_dim,
            'num_dim': num_dim,
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        
        # Print final results
        final_train_metric = train_metrics[-1]
        final_val_metric = val_metrics[-1]
        
        print(f"\n📊 Final Results for {model_name} ({task_type}):")
        print(f"   Training - Loss: {train_losses[-1]:.4f}, Metrics: {final_train_metric}")
        print(f"   Validation - Loss: {val_losses[-1]:.4f}, Metrics: {final_val_metric}")
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'final_val_metrics': final_val_metric,
            'model_path': model_path
        }
    
    def train_all_models(self, df: pd.DataFrame, task_types: List[str] = None, 
                        model_names: List[str] = None) -> Dict:
        """Train all models for all tasks with comprehensive reporting"""
        if task_types is None:
            task_types = ['classification', 'regression']
        if model_names is None:
            model_names = ['EarlyFusion', 'LateFusion', 'HybridFusion', 'CrossModalTransformer', 
                          'EnsembleFusion', 'AdaptiveEnsemble', 'LSTM', 'Transformer']
        
        print(f"\n🚀 Starting comprehensive model training...")
        print(f"📋 Tasks: {task_types}")
        print(f"🤖 Models: {model_names}")
        print("=" * 80)
        
        results = {}
        
        for task_type in task_types:
            print(f"\n🎯 Training models for {task_type.upper()} task...")
            print("-" * 60)
            results[task_type] = {}
            
            for model_name in model_names:
                try:
                    result = self.train_model(model_name, df, task_type)
                    results[task_type][model_name] = result
                    self.trained_models[f"{model_name}_{task_type}"] = result['model']
                    self.model_metrics[f"{model_name}_{task_type}"] = result['final_val_metrics']
                    self.training_history[f"{model_name}_{task_type}"] = {
                        'train_losses': result['train_losses'],
                        'val_losses': result['val_losses'],
                        'train_metrics': result['train_metrics'],
                        'val_metrics': result['val_metrics']
                    }
                    print(f"✅ {model_name} completed successfully!")
                except Exception as e:
                    import traceback
                    error_msg = f"Error training {model_name} for {task_type}: {str(e)}"
                    print(f"❌ {error_msg}")
                    print(f"📋 Full error traceback:")
                    print(traceback.format_exc())
                    results[task_type][model_name] = {'error': str(e)}
        
        return results
    
    def plot_training_results(self, save_path: str = "reports/training_results.png"):
        """Create comprehensive training visualization"""
        if not self.training_history:
            print("❌ No training history available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Training Results', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        for idx, (model_key, history) in enumerate(self.training_history.items()):
            if idx >= 4:
                break
                
            ax = axes[idx]
            
            # Plot losses
            epochs = range(1, len(history['train_losses']) + 1)
            ax.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
            
            ax.set_title(f'{model_key}', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add metrics as text
            final_metrics = self.model_metrics.get(model_key, {})
            metrics_text = '\n'.join([f"{k}: {v:.4f}" for k, v in final_metrics.items()])
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training plots saved to: {save_path}")
        plt.show()
    
    def generate_report(self, results: Dict, save_path: str = "reports/model_performance_report.txt"):
        """Generate comprehensive performance report"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MULTI-MODAL REVIEW ANALYZER - MODEL PERFORMANCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Training completed on device: {DEVICE}\n")
            f.write(f"SBERT Model: {SBERT_MODEL_NAME}\n")
            f.write(f"Total models trained: {len(self.trained_models)}\n\n")
            
            for task_type, task_results in results.items():
                f.write(f"{task_type.upper()} TASK RESULTS\n")
                f.write("-" * 40 + "\n")
                
                for model_name, result in task_results.items():
                    f.write(f"\n{model_name}:\n")
                    if 'error' in result:
                        f.write(f"  ERROR: {result['error']}\n")
                    else:
                        f.write(f"  SUCCESS: Successfully trained\n")
                        f.write(f"  Final validation metrics:\n")
                        for metric, value in result['final_val_metrics'].items():
                            f.write(f"    - {metric}: {value:.4f}\n")
                        f.write(f"  Model saved to: {result['model_path']}\n")
                
                f.write("\n" + "=" * 40 + "\n")
        
        print(f"📄 Performance report saved to: {save_path}")
    
    def predict(self, model_name: str, task_type: str, text: str) -> Dict:
        """Make prediction using trained model"""
        model_key = f"{model_name}_{task_type}"
        if model_key not in self.trained_models:
            return {'error': 'Model not trained'}
        
        try:
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(text)
            filtered_text = self.preprocessor.remove_stopwords(cleaned_text)
            
            # Get embedding
            embedding = self.encoder.encode([filtered_text], max_length=MAX_SEQ_LENGTH)
            X_text = torch.tensor(embedding, dtype=torch.float32, device=DEVICE)
            
            # Get sentiment features
            sent_feats = sentiment_features(filtered_text)
            X_num = torch.tensor([[sent_feats['polarity'], sent_feats['subjectivity']]], 
                                dtype=torch.float32, device=DEVICE)
            
            # Predict
            model = self.trained_models[model_key]
            model.eval()
            with torch.no_grad():
                output = model(X_text, X_num)
                
                if task_type == 'classification':
                    probs = torch.softmax(output, dim=1)
                    pred_class = output.argmax(dim=1).item()
                    confidence = probs[0][pred_class].item()
                    return {
                        'prediction': 'Positive' if pred_class == 1 else 'Negative',
                        'confidence': confidence,
                        'probabilities': probs[0].cpu().numpy().tolist()
                    }
                else:
                    rating = output.squeeze().item()
                    return {
                        'predicted_rating': rating,
                        'rounded_rating': round(rating, 1)
                    }
                    
        except Exception as e:
            return {'error': str(e)}
    
    def generate_advanced_analysis(self, results: Dict):
        """Generate advanced model comparison and analysis"""
        print("🔬 Generating advanced model analysis...")
        
        # Collect all model metrics for comparison
        all_metrics = {}
        for task_type, task_results in results.items():
            all_metrics[task_type] = {}
            for model_name, result in task_results.items():
                if 'error' not in result:
                    all_metrics[task_type][model_name] = result['final_val_metrics']
        
        # Generate advanced comparison
        for task_type, metrics in all_metrics.items():
            if metrics:
                print(f"\n📊 Advanced Analysis for {task_type.upper()}:")
                comparison = advanced_model_comparison(metrics)
                
                print(f"🏆 Best models by metric:")
                for metric, data in comparison.items():
                    if metric not in ['overall_ranking', 'model_scores']:
                        print(f"   - {metric.replace('_', ' ').title()}: {data['best_model']} ({data['best_value']:.4f})")
                
                print(f"\n🥇 Overall ranking:")
                for i, (model, score) in enumerate(comparison['overall_ranking'][:3], 1):
                    print(f"   {i}. {model}: {score:.4f}")
        
        # Generate comprehensive report
        report_path = "reports/advanced_model_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ADVANCED MODEL ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for task_type, metrics in all_metrics.items():
                if metrics:
                    f.write(f"{task_type.upper()} TASK - ADVANCED ANALYSIS\n")
                    f.write("-" * 50 + "\n")
                    
                    comparison = advanced_model_comparison(metrics)
                    
                    f.write("Best Models by Metric:\n")
                    for metric, data in comparison.items():
                        if metric not in ['overall_ranking', 'model_scores']:
                            f.write(f"  {metric.replace('_', ' ').title()}: {data['best_model']} ({data['best_value']:.4f})\n")
                    
                    f.write("\nOverall Ranking:\n")
                    for i, (model, score) in enumerate(comparison['overall_ranking'], 1):
                        f.write(f"  {i}. {model}: {score:.4f}\n")
                    
                    f.write("\n" + "=" * 50 + "\n")
        
        print(f"📄 Advanced analysis report saved to: {report_path}")
    
    def generate_cross_modal_analysis(self, results: Dict):
        """Generate cross-modal analysis and visualizations"""
        print("🔬 Generating cross-modal analysis...")
        
        try:
            # Collect fusion model results
            fusion_results = {}
            for task_type, task_results in results.items():
                fusion_results[task_type] = {}
                fusion_models = ['EarlyFusion', 'LateFusion', 'HybridFusion', 'CrossModalTransformer']
                for model_name in fusion_models:
                    if model_name in task_results and 'error' not in task_results[model_name]:
                        fusion_results[task_type][model_name] = task_results[model_name]['final_val_metrics']
            
            # Generate fusion comparison visualization
            if fusion_results:
                try:
                    fusion_fig = create_fusion_technique_comparison(fusion_results)
                    fusion_fig.write_html("reports/fusion_technique_comparison.html")
                    print("📊 Fusion technique comparison saved to: reports/fusion_technique_comparison.html")
                except Exception as e:
                    print(f"⚠️  Warning: Could not generate fusion comparison chart: {e}")
                    # Create a simple comparison chart instead
                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        for task_type, models in fusion_results.items():
                            for model_name, metrics in models.items():
                                fig.add_trace(go.Bar(
                                    name=f"{model_name}_{task_type}",
                                    x=[f"{model_name}_{task_type}"],
                                    y=[metrics.get('Accuracy', metrics.get('MAE', 0))]
                                ))
                        fig.update_layout(title="Fusion Model Performance Comparison")
                        fig.write_html("reports/fusion_technique_comparison.html")
                        print("📊 Simple fusion comparison saved to: reports/fusion_technique_comparison.html")
                    except Exception as e2:
                        print(f"⚠️  Could not create simple comparison chart: {e2}")
            
            # Generate model comparison charts for each task
            for task_type, task_results in results.items():
                if task_results:
                    # Collect metrics for visualization
                    model_metrics = {}
                    for model_name, result in task_results.items():
                        if 'error' not in result:
                            model_metrics[model_name] = result['final_val_metrics']
                    
                    if model_metrics:
                        try:
                            # Create comparison chart
                            comparison_fig = create_model_comparison_chart(model_metrics, task_type)
                            comparison_fig.write_html(f"reports/{task_type}_model_comparison.html")
                            
                            # Create radar chart
                            radar_fig = create_performance_radar_chart(model_metrics)
                            radar_fig.write_html(f"reports/{task_type}_performance_radar.html")
                            
                            print(f"📊 {task_type.title()} analysis charts saved to reports/")
                        except Exception as e:
                            print(f"⚠️  Warning: Could not generate {task_type} visualization: {e}")
            
            # Generate interactive HTML dashboard
            all_model_metrics = {}
            for task_type, task_results in results.items():
                for model_name, result in task_results.items():
                    if 'error' not in result:
                        all_model_metrics[f"{model_name}_{task_type}"] = result['final_val_metrics']
            
            if all_model_metrics:
                try:
                    html_filename = export_results_to_html(all_model_metrics, "reports/multimodal_analysis_dashboard.html")
                    print(f"🎨 Interactive dashboard saved to: {html_filename}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not generate interactive dashboard: {e}")
        
        except Exception as e:
            print(f"⚠️  Error generating cross-modal analysis: {e}")
    
    def enhanced_predict(self, model_name: str, task_type: str, text: str) -> Dict:
        """Enhanced prediction with cross-modal feature analysis"""
        model_key = f"{model_name}_{task_type}"
        if model_key not in self.trained_models:
            return {'error': 'Model not trained'}
        
        try:
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(text)
            filtered_text = self.preprocessor.remove_stopwords(cleaned_text)
            
            # Get embedding
            embedding = self.encoder.encode([filtered_text], max_length=MAX_SEQ_LENGTH)
            X_text = torch.tensor(embedding, dtype=torch.float32, device=DEVICE)
            
            # Get advanced sentiment features
            try:
                sent_feats = advanced_sentiment_features(filtered_text)
                linguistic_feats = extract_linguistic_features(filtered_text)
            except Exception as e:
                print(f"⚠️  Warning: Error in feature extraction: {e}")
                # Fallback to basic features
                sent_feats = {
                    'polarity': 0.0, 'subjectivity': 0.0, 'word_count': len(filtered_text.split()),
                    'char_count': len(filtered_text), 'avg_word_length': 0.0,
                    'sentiment_intensity': 0.0, 'sentiment_confidence': 0.0
                }
                linguistic_feats = {
                    'word_count': len(filtered_text.split()), 'char_count': len(filtered_text),
                    'avg_word_length': 0.0, 'punctuation_ratio': 0.0, 'uppercase_ratio': 0.0,
                    'lexical_diversity': 0.0, 'positive_word_count': 0, 'negative_word_count': 0,
                    'sentiment_lexicon_score': 0.0
                }
            
            # Mock numerical features for cross-modal analysis
            mock_num_feats = {
                'rating': 4.0,  # Default rating
                'helpful_votes': 5,  # Default helpful votes
                'review_count': 1
            }
            
            # Create cross-modal features
            cross_modal_feats = cross_modal_feature_engineering(sent_feats, mock_num_feats)
            
            # Combine features
            num_features = [sent_feats['polarity'], sent_feats['subjectivity'], 
                          sent_feats['word_count'], sent_feats['char_count'], 
                          sent_feats['avg_word_length']]
            X_num = torch.tensor([num_features], dtype=torch.float32, device=DEVICE)
            
            # Predict
            model = self.trained_models[model_key]
            model.eval()
            with torch.no_grad():
                output = model(X_text, X_num)
                
                if task_type == 'classification':
                    probs = torch.softmax(output, dim=1)
                    pred_class = output.argmax(dim=1).item()
                    confidence = probs[0][pred_class].item()
                    return {
                        'prediction': 'Positive' if pred_class == 1 else 'Negative',
                        'confidence': confidence,
                        'probabilities': probs[0].cpu().numpy().tolist(),
                        'sentiment_features': sent_feats,
                        'linguistic_features': linguistic_feats,
                        'cross_modal_features': cross_modal_feats
                    }
                else:
                    rating = output.squeeze().item()
                    return {
                        'predicted_rating': rating,
                        'rounded_rating': round(rating, 1),
                        'sentiment_features': sent_feats,
                        'linguistic_features': linguistic_feats,
                        'cross_modal_features': cross_modal_feats
                    }
                    
        except Exception as e:
            return {'error': str(e)}


def main():
    """Main training function with command-line arguments"""
    parser = argparse.ArgumentParser(description="Train multimodal review analyzer models")
    parser.add_argument("--models", nargs="+", 
                       default=['EarlyFusion', 'LateFusion', 'HybridFusion', 'CrossModalTransformer'],
                       help="Models to train")
    parser.add_argument("--tasks", nargs="+", default=['classification', 'regression'],
                       help="Tasks to train for")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--max-rows", type=int, default=15000,
                       help="Maximum number of rows to use for training")
    parser.add_argument("--data-file", type=str, default=None,
                       help="Specific data file to use")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip advanced analysis generation")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with detailed logging")
    
    args = parser.parse_args()
    
    print("🚀 Multi-Modal Review Analyzer - Enhanced Training Pipeline")
    print("=" * 70)
    print(f"📋 Configuration:")
    print(f"   - Models: {args.models}")
    print(f"   - Tasks: {args.tasks}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Max rows: {args.max_rows}")
    print(f"   - Skip analysis: {args.skip_analysis}")
    print(f"   - Debug mode: {args.debug}")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    print("\n📁 Loading data...")
    try:
        # Get available files
        raw_files = list_jsonl_files(RAW_DATA_DIR)
        if not raw_files:
            print("❌ No JSONL files found in data/raw/")
            return
        
        # Use specified file or first available file or default
        if args.data_file:
            reviews_file = args.data_file
            selected_reviews_path = os.path.join(RAW_DATA_DIR, reviews_file)
            if not os.path.exists(selected_reviews_path):
                print(f"❌ Specified data file not found: {selected_reviews_path}")
                return
        else:
            reviews_file = raw_files[0] if raw_files else os.path.basename(REVIEWS_FILE)
            selected_reviews_path = os.path.join(RAW_DATA_DIR, reviews_file)
        
        print(f"📄 Using reviews file: {reviews_file}")
        
        # Process and load data
        parquet_path = ensure_processed_parquet(selected_reviews_path)
        reviews_df = load_processed_parquet(parquet_path)
        
        print(f"✅ Data loaded successfully: {len(reviews_df)} samples")
        
        if args.debug:
            print(f"📋 Data columns: {list(reviews_df.columns)}")
            print(f"📋 Data types: {reviews_df.dtypes.to_dict()}")
            print(f"📋 Sample data (first 3 rows):")
            print(reviews_df.head(3))
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Use command-line arguments
    task_types = args.tasks
    model_names = args.models
    epochs = args.epochs
    max_rows = args.max_rows
    
    # Train all models
    results = trainer.train_all_models(reviews_df, task_types, model_names)
    
    # Generate visualizations and reports
    print("\n📊 Generating reports and visualizations...")
    trainer.plot_training_results()
    trainer.generate_report(results)
    
    # Generate advanced analysis and comparisons (unless skipped)
    if not args.skip_analysis:
        print("\n🔬 Generating advanced analysis...")
        trainer.generate_advanced_analysis(results)
        trainer.generate_cross_modal_analysis(results)
    else:
        print("\n⏭️  Skipping advanced analysis generation")
    
    # Summary
    print("\n🎉 Training completed successfully!")
    print(f"📊 Trained models: {len(trainer.trained_models)}")
    print(f"📈 Performance metrics:")
    for model_key, metrics in trainer.model_metrics.items():
        print(f"   - {model_key}: {metrics}")
    
    # Test inference with enhanced features
    print("\n🧪 Testing enhanced inference...")
    test_text = "This is an amazing product! I love it and would definitely recommend it to others."
    
    # Test with enhanced prediction (shows cross-modal features)
    print("🔬 Enhanced prediction with cross-modal analysis:")
    for model_key in list(trainer.trained_models.keys())[:2]:  # Test first 2 models
        model_name, task_type = model_key.split('_')
        result = trainer.enhanced_predict(model_name, task_type, test_text)
        if 'error' not in result:
            print(f"\n   📊 {model_key} Results:")
            if task_type == 'classification':
                print(f"      - Prediction: {result['prediction']}")
                print(f"      - Confidence: {result['confidence']:.3f}")
            else:
                print(f"      - Predicted Rating: {result['predicted_rating']:.2f}")
                print(f"      - Rounded Rating: {result['rounded_rating']}")
            
            print(f"      - Sentiment Polarity: {result['sentiment_features']['polarity']:.3f}")
            print(f"      - Word Count: {result['sentiment_features']['word_count']}")
            print(f"      - Cross-modal Features: {len(result['cross_modal_features'])} computed")
        else:
            print(f"   - {model_key}: Error - {result['error']}")
    
    # Test standard inference for comparison
    print("\n🧪 Standard inference comparison:")
    for model_key in list(trainer.trained_models.keys())[:2]:
        model_name, task_type = model_key.split('_')
        result = trainer.predict(model_name, task_type, test_text)
        if 'error' not in result:
            print(f"   - {model_key}: {result}")
        else:
            print(f"   - {model_key}: Error - {result['error']}")


if __name__ == "__main__":
    main()

