import os
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import streamlit as st
from pathlib import Path

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
from models.fusion_model import EarlyFusionModel, HybridFusionModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerEncoderModel
from preprocessing.feature_engineering import sentiment_features
from utils.metrics import classification_metrics, regression_metrics
from utils.visualization import plot_rating_distribution

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)

class ModelTrainer:
    """Comprehensive model trainer and manager"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.encoder = SBERTEncoder(model_name=SBERT_MODEL_NAME, device=DEVICE)
        self.trained_models = {}
        self.model_metrics = {}
        
    def get_embedding_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith('emb_')]
    
    def build_dataset_from_df(self, df: pd.DataFrame, task_type: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build dataset from processed DataFrame"""
        emb_cols = self.get_embedding_columns(df)
        if not emb_cols:
            raise ValueError("No embedding columns found in data")
            
        X_text = torch.tensor(df[emb_cols].values, dtype=torch.float32, device=DEVICE)
        
        # Numerical features
        num_feats = ['sent_polarity', 'sent_subjectivity']
        for nf in num_feats:
            if nf not in df.columns:
                df[nf] = 0.0
        X_num = torch.tensor(df[num_feats].values, dtype=torch.float32, device=DEVICE)
        
        # Labels
        if task_type == 'classification':
            if 'overall' in df.columns:
                y = (df['overall'] >= 4).astype(int).values
            else:
                y = np.random.randint(0, 2, len(df))
            y = torch.tensor(y, dtype=torch.long, device=DEVICE)
        else:  # regression
            if 'overall' in df.columns:
                y = torch.tensor(df['overall'].values, dtype=torch.float32, device=DEVICE)
            else:
                y = torch.randn(len(df), dtype=torch.float32, device=DEVICE)
                
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
        """Create model based on name and task"""
        hidden_dim = 256
        
        if task_type == 'classification':
            output_dim = 2
        else:
            output_dim = 1
            
        if model_name == 'EarlyFusion':
            return EarlyFusionModel(text_dim, num_dim, hidden_dim, output_dim).to(DEVICE)
            
        elif model_name == 'HybridFusion':
            return HybridFusionModel(text_dim, num_dim, hidden_dim, output_dim).to(DEVICE)
            
        elif model_name == 'LSTM':
            seq_dim = self.prepare_sequence_from_embeddings(
                torch.zeros(1, text_dim, device=DEVICE)
            ).shape[2]
            
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
            return wrapper
            
        elif model_name == 'Transformer':
            seq_dim = self.prepare_sequence_from_embeddings(
                torch.zeros(1, text_dim, device=DEVICE)
            ).shape[2]
            
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
            return wrapper
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_model(self, model_name: str, df: pd.DataFrame, task_type: str, 
                   epochs: int = NUM_EPOCHS, max_rows: int = 10000) -> Dict:
        """Train a single model"""
        print(f"Training {model_name} for {task_type} task...")
        
        # Subsample for faster training
        if len(df) > max_rows:
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
        
        # Training loop
        train_losses, val_losses = [], []
        train_metrics, val_metrics = [], []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []
            
            for i in range(0, len(train_idx), BATCH_SIZE):
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
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            
            with torch.no_grad():
                for i in range(0, len(val_idx), BATCH_SIZE):
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
            
            # Calculate metrics
            if task_type == 'classification':
                train_metric = classification_metrics(train_targets, train_preds)
                val_metric = classification_metrics(val_targets, val_preds)
            else:
                train_metric = regression_metrics(train_targets, train_preds)
                val_metric = regression_metrics(val_targets, val_preds)
            
            train_losses.append(train_loss / len(train_idx) * BATCH_SIZE)
            val_losses.append(val_loss / len(val_idx) * BATCH_SIZE)
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, "
                  f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_metric.get('Accuracy', val_metric.get('MAE', 0)):.4f}")
        
        # Save model
        model_path = os.path.join(os.path.dirname(SAVED_MODEL_PATH), f"{model_name}_{task_type}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'task_type': task_type,
            'text_dim': text_dim,
            'num_dim': num_dim,
            'epochs': epochs
        }, model_path)
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'final_val_metrics': val_metrics[-1],
            'model_path': model_path
        }
    
    def train_all_models(self, df: pd.DataFrame, task_types: List[str] = None, 
                        model_names: List[str] = None) -> Dict:
        """Train all models for all tasks"""
        if task_types is None:
            task_types = ['classification', 'regression']
        if model_names is None:
            model_names = ['EarlyFusion', 'HybridFusion', 'LSTM', 'Transformer']
        
        results = {}
        
        for task_type in task_types:
            results[task_type] = {}
            for model_name in model_names:
                try:
                    result = self.train_model(model_name, df, task_type)
                    results[task_type][model_name] = result
                    self.trained_models[f"{model_name}_{task_type}"] = result['model']
                    self.model_metrics[f"{model_name}_{task_type}"] = result['final_val_metrics']
                except Exception as e:
                    print(f"Error training {model_name} for {task_type}: {e}")
                    results[task_type][model_name] = {'error': str(e)}
        
        return results
    
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

# Streamlit App
def create_streamlit_app():
    """Create the Streamlit application"""
    
    # Vercel-inspired styling
    st.markdown(
        """
        <style>
        :root { 
            --bg:#0b0f19; --card:#0f1525; --muted:#9aa4b2; --text:#e6e8eb; 
            --primary:#7c3aed; --accent:#22d3ee; --success:#10b981; 
        }
        .stApp { 
            background: linear-gradient(180deg, var(--bg), #0b0f19 60%); 
            color: var(--text); 
        }
        header { background: transparent; }
        .block-container { padding-top: 1.5rem; }
        .vercel-card { 
            background: var(--card); 
            border-radius: 16px; 
            padding: 20px; 
            border: 1px solid #1f2637; 
            margin-bottom: 20px;
        }
        .metric { 
            background: var(--card); 
            padding: 16px; 
            border-radius: 12px; 
            border: 1px solid #1f2637; 
            text-align: center;
        }
        .section-title { 
            font-size: 0.9rem; 
            color: var(--muted); 
            text-transform: uppercase; 
            letter-spacing: 0.08em; 
            margin-bottom: 8px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 12px; }
        .stTabs [data-baseweb="tab"] { 
            background: var(--card); 
            color: var(--text); 
            border-radius: 10px; 
            padding: 8px 16px; 
            border: 1px solid #1f2637; 
        }
        .stTabs [aria-selected="true"] { border-color: var(--accent); }
        .success-box { 
            background: rgba(16, 185, 129, 0.1); 
            border: 1px solid var(--success); 
            border-radius: 8px; 
            padding: 12px; 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.set_page_config(
        page_title="Multi-Modal Review Analyzer",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("✨ Multi-Modal Review Analyzer")
    
    # Initialize trainer
    if 'trainer' not in st.session_state:
        st.session_state.trainer = ModelTrainer()
    
    # Sidebar
    st.sidebar.header("📁 Data Files")
    raw_files = list_jsonl_files(RAW_DATA_DIR)
    default_reviews = os.path.basename(REVIEWS_FILE)
    default_metadata = os.path.basename(METADATA_FILE)
    
    if default_reviews not in raw_files and os.path.exists(REVIEWS_FILE):
        raw_files.append(default_reviews)
    if default_metadata not in raw_files and os.path.exists(METADATA_FILE):
        raw_files.append(default_metadata)
    raw_files = sorted(set(raw_files))
    
    reviews_choice = st.sidebar.selectbox(
        "Select reviews JSONL",
        options=raw_files,
        index=raw_files.index(default_reviews) if default_reviews in raw_files else 0,
        key="reviews_file_choice",
    ) if raw_files else None
    
    selected_reviews_path = os.path.join(RAW_DATA_DIR, reviews_choice) if reviews_choice else REVIEWS_FILE
    
    # Process and load data
    if st.sidebar.button("🔄 Process Data", type="primary"):
        with st.spinner("Processing data..."):
            parquet_path = ensure_processed_parquet(selected_reviews_path)
            st.success("Data processed successfully!")
    
    # Load processed data
    try:
        parquet_path = ensure_processed_parquet(selected_reviews_path)
        reviews_df = load_processed_parquet(parquet_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Main tabs
    tabs = st.tabs(["📊 Overview", "🔍 Explore", "🚀 Train Models", "🎯 Inference"])
    
    with tabs[0]:  # Overview
        st.markdown("<div class='vercel-card'>", unsafe_allow_html=True)
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class='metric'>
                <div class='section-title'>Total Reviews</div>
                <h2>{len(reviews_df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            emb_cols = st.session_state.trainer.get_embedding_columns(reviews_df)
            st.markdown(f"""
            <div class='metric'>
                <div class='section-title'>Embedding Dim</div>
                <h2>{len(emb_cols)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with c3:
            if 'overall' in reviews_df.columns:
                avg_rating = reviews_df['overall'].mean()
                st.markdown(f"""
                <div class='metric'>
                    <div class='section-title'>Avg Rating</div>
                    <h2>{avg_rating:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='metric'>
                    <div class='section-title'>Avg Rating</div>
                    <h2>N/A</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with c4:
            if 'sent_polarity' in reviews_df.columns:
                avg_sentiment = reviews_df['sent_polarity'].mean()
                st.markdown(f"""
                <div class='metric'>
                    <div class='section-title'>Avg Sentiment</div>
                    <h2>{avg_sentiment:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='metric'>
                    <div class='section-title'>Avg Sentiment</div>
                    <h2>N/A</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Rating distribution
        st.subheader("📈 Rating Distribution")
        if "overall" in reviews_df.columns:
            plot_rating_distribution(reviews_df, rating_column="overall")
        else:
            st.info("No 'overall' column found in dataset.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[1]:  # Explore
        st.markdown("<div class='vercel-card'>", unsafe_allow_html=True)
        st.subheader("🔍 Data Exploration")
        
        # Sample data
        st.subheader("📋 Sample Data")
        if len(reviews_df) > 0:
            st.dataframe(reviews_df.head(10))
        else:
            st.warning("No data available")
        
        # Text preprocessing example
        if len(reviews_df) > 0 and 'reviewText' in reviews_df.columns:
            st.subheader("🧹 Text Preprocessing Example")
            example_text = reviews_df['reviewText'].iloc[0]
            cleaned_text = st.session_state.trainer.preprocessor.clean_text(example_text)
            filtered_text = st.session_state.trainer.preprocessor.remove_stopwords(cleaned_text)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Original:**")
                st.text(example_text)
            with col2:
                st.write("**Cleaned:**")
                st.text(cleaned_text)
            with col3:
                st.write("**No Stopwords:**")
                st.text(filtered_text)
        
        # Embedding preview
        if len(reviews_df) > 0 and 'reviewText' in reviews_df.columns:
            st.subheader("🧠 SBERT Embedding Preview")
            try:
                example_text = reviews_df['reviewText'].iloc[0]
                cleaned_text = st.session_state.trainer.preprocessor.clean_text(example_text)
                filtered_text = st.session_state.trainer.preprocessor.remove_stopwords(cleaned_text)
                embedding = st.session_state.trainer.encoder.encode([filtered_text], max_length=MAX_SEQ_LENGTH)
                st.write(f"**Embedding Shape:** {embedding.shape}")
                st.write(f"**First 10 dimensions:** {embedding[0][:10].tolist()}")
            except Exception as e:
                st.warning(f"Could not compute embedding: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[2]:  # Train Models
        st.markdown("<div class='vercel-card'>", unsafe_allow_html=True)
        st.subheader("🚀 Model Training")
        
        # Training controls
        col1, col2, col3 = st.columns(3)
        with col1:
            task_types = st.multiselect(
                "Select Tasks", 
                ["classification", "regression"], 
                default=["classification"]
            )
        with col2:
            model_names = st.multiselect(
                "Select Models", 
                ["EarlyFusion", "HybridFusion", "LSTM", "Transformer"], 
                default=["EarlyFusion", "HybridFusion"]
            )
        with col3:
            epochs = st.slider("Epochs", 1, 20, 5)
            max_rows = st.slider("Max Training Rows", 1000, 50000, 10000)
        
        if st.button("🎯 Start Training", type="primary"):
            if not task_types or not model_names:
                st.error("Please select at least one task and one model")
            else:
                with st.spinner("Training models... This may take a while."):
                    try:
                        results = st.session_state.trainer.train_all_models(
                            reviews_df, task_types, model_names
                        )
                        
                        st.success("🎉 Training completed successfully!")
                        
                        # Display results
                        for task_type in task_types:
                            st.subheader(f"📊 {task_type.title()} Results")
                            for model_name in model_names:
                                if model_name in results[task_type]:
                                    result = results[task_type][model_name]
                                    if 'error' in result:
                                        st.error(f"❌ {model_name}: {result['error']}")
                                    else:
                                        metrics = result['final_val_metrics']
                                        st.write(f"✅ **{model_name}**: {metrics}")
                                        
                    except Exception as e:
                        st.error(f"Training failed: {e}")
        
        # Show trained models
        if st.session_state.trainer.trained_models:
            st.subheader("🎯 Trained Models")
            for model_key, metrics in st.session_state.trainer.model_metrics.items():
                st.write(f"**{model_key}**: {metrics}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[3]:  # Inference
        st.markdown("<div class='vercel-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Model Inference")
        
        if not st.session_state.trainer.trained_models:
            st.warning("⚠️ No trained models available. Please train models first.")
        else:
            # Model selection
            col1, col2 = st.columns(2)
            with col1:
                available_models = list(st.session_state.trainer.trained_models.keys())
                selected_model = st.selectbox("Select Model", available_models)
            with col2:
                st.write(f"**Metrics:** {st.session_state.trainer.model_metrics.get(selected_model, 'N/A')}")
            
            # Text input
            user_text = st.text_area(
                "Enter review text:", 
                "This is an amazing product! I love it and would definitely recommend it to others.",
                height=100
            )
            
            if st.button("🔮 Make Prediction", type="primary"):
                if user_text.strip():
                    model_name, task_type = selected_model.split('_')
                    result = st.session_state.trainer.predict(model_name, task_type, user_text)
                    
                    if 'error' in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.success("✅ Prediction completed!")
                        
                        if task_type == 'classification':
                            st.write(f"**Prediction:** {result['prediction']}")
                            st.write(f"**Confidence:** {result['confidence']:.3f}")
                            
                            # Show probabilities
                            st.write("**Class Probabilities:**")
                            probs = result['probabilities']
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Negative", f"{probs[0]:.3f}")
                            with col2:
                                st.metric("Positive", f"{probs[1]:.3f}")
                        else:
                            st.write(f"**Predicted Rating:** {result['predicted_rating']:.2f}")
                            st.write(f"**Rounded Rating:** {result['rounded_rating']}")
                else:
                    st.warning("Please enter some text to analyze.")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    create_streamlit_app()
