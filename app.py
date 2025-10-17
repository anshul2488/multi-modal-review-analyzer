import os
import streamlit as st
import pandas as pd
import torch

from config import (
    REVIEWS_FILE, METADATA_FILE, DEVICE, SBERT_MODEL_NAME, MAX_SEQ_LENGTH, RAW_DATA_DIR
)
from utils.data_loader import (
    load_jsonl,
    list_jsonl_files,
    ensure_processed_parquet,
    load_processed_parquet,
)
from preprocessing.text_preprocessor import TextPreprocessor
from models.nlp_utils import SBERTEncoder
from utils.visualization import (
    plot_rating_distribution, create_model_comparison_chart, 
    create_fusion_technique_comparison, create_cross_modal_analysis_chart,
    create_performance_radar_chart, export_results_to_html
)
from utils.metrics import (
    classification_metrics, regression_metrics, cross_modal_evaluation_metrics,
    advanced_model_comparison, generate_model_report
)
from preprocessing.feature_engineering import (
    advanced_sentiment_features, cross_modal_feature_engineering,
    extract_linguistic_features
)
from models.fusion_model import EarlyFusionModel, HybridFusionModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerEncoderModel

st.set_page_config(
    page_title="Multi-Modal Review Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure caching for better performance
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_data_info():
    return {"cached": True}

# Vercel-inspired styling
st.markdown(
    """
    <style>
    :root { --bg:#0b0f19; --card:#0f1525; --muted:#9aa4b2; --text:#e6e8eb; --primary:#7c3aed; --accent:#22d3ee; }
    .stApp { background: linear-gradient(180deg, var(--bg), #0b0f19 60%); color: var(--text); }
    header { background: transparent; }
    .block-container { padding-top: 1.5rem; }
    .vercel-card { background: var(--card); border-radius: 16px; padding: 18px; border: 1px solid #1f2637; }
    .metric { background: var(--card); padding: 12px 14px; border-radius: 12px; border: 1px solid #1f2637; }
    .section-title { font-size: 0.9rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { background: var(--card); color: var(--text); border-radius: 10px; padding: 8px 12px; border: 1px solid #1f2637; }
    .stTabs [aria-selected="true"] { border-color: var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=3600, show_spinner=True)
def load_processed_data(parquet_path: str, sample_size: int = 100000):
    """Load processed parquet data for fast access with optional sampling"""
    try:
        if os.path.exists(parquet_path):
            # Check file size and load accordingly
            file_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB
            
            if file_size > 100:  # If file is larger than 100MB, sample directly
                df = pd.read_parquet(parquet_path)
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            else:
                df = pd.read_parquet(parquet_path)
            
            # Map column names to match expected format
            if 'rating' in df.columns:
                df = df.rename(columns={'rating': 'overall'})
            if 'text' in df.columns:
                df = df.rename(columns={'text': 'reviewText'})
            return df
        else:
            st.warning(f"Processed data file not found: {parquet_path}")
            return pd.DataFrame()
    except Exception as exc:
        st.error(f"Failed to load processed data: {exc}")
        return pd.DataFrame()

@st.cache_data(show_spinner=True)
def load_raw_data(reviews_path: str, metadata_path: str):
    """Load raw data for metadata only (if needed)"""
    metadata_df = pd.DataFrame()
    try:
        if os.path.exists(metadata_path):
            metadata_df = load_jsonl(metadata_path)
        else:
            st.warning(f"Metadata file not found: {metadata_path}")
    except Exception as exc:
        st.error(f"Failed to load metadata: {exc}")
    return metadata_df

@st.cache_resource(show_spinner=True)
def get_preprocessor():
    return TextPreprocessor()

@st.cache_resource(show_spinner=True)
def get_encoder():
    return SBERTEncoder(model_name=SBERT_MODEL_NAME, device=DEVICE)

def get_embedding_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith('emb_')]

def build_dataset_from_df(df: pd.DataFrame, task_type: str):
    emb_cols = get_embedding_columns(df)
    if not emb_cols:
        return None, None, None
    X_text = torch.tensor(df[emb_cols].values, dtype=torch.float32, device=DEVICE)
    # numerical features from processing
    num_feats = ['sent_polarity', 'sent_subjectivity']
    for nf in num_feats:
        if nf not in df.columns:
            df[nf] = 0.0
    X_num = torch.tensor(df[num_feats].values, dtype=torch.float32, device=DEVICE)
    if task_type == 'classification':
        # label: positive if rating >= 4 else negative
        if 'overall' in df.columns:
            y = (df['overall'] >= 4).astype(int).values
            y = torch.tensor(y, dtype=torch.long, device=DEVICE)
        else:
            y = torch.zeros(len(df), dtype=torch.long, device=DEVICE)
    else:
        # regression on rating if available
        if 'overall' in df.columns:
            y = torch.tensor(df['overall'].values, dtype=torch.float32, device=DEVICE)
        else:
            y = torch.zeros(len(df), dtype=torch.float32, device=DEVICE)
    return X_text, X_num, y

def _prepare_sequence_from_embeddings(X_text: torch.Tensor, seq_len: int = 16):
    n, d = X_text.shape
    if d % seq_len != 0:
        new_d = ((d + seq_len - 1) // seq_len) * seq_len
        pad = new_d - d
        X_text = torch.nn.functional.pad(X_text, (0, pad))
        d = new_d
    step = d // seq_len
    return X_text.view(n, seq_len, step)

def train_model(df: pd.DataFrame, model_name: str, task_type: str, max_rows: int = 5000, epochs: int = 3):
    # Subsample for speed
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    X_text, X_num, y = build_dataset_from_df(df, task_type)
    if X_text is None:
        return None, {}
    text_dim = X_text.shape[1]
    num_dim = X_num.shape[1]
    hidden_dim = 256
    if task_type == 'classification':
        output_dim = 2
        criterion = torch.nn.CrossEntropyLoss()
    else:
        output_dim = 1
        criterion = torch.nn.MSELoss()

    if model_name == 'EarlyFusion':
        model = EarlyFusionModel(text_dim=text_dim, num_dim=num_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)
        forward = lambda xb_text, xb_num: model(xb_text, xb_num)
    elif model_name == 'HybridFusion':
        model = HybridFusionModel(text_dim=text_dim, num_dim=num_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(DEVICE)
        forward = lambda xb_text, xb_num: model(xb_text, xb_num)
    elif model_name == 'LSTM':
        base = LSTMModel(input_size=_prepare_sequence_from_embeddings(X_text).shape[2], hidden_size=192, num_layers=1, output_size=hidden_dim, dropout=0.2).to(DEVICE)
        head = torch.nn.Linear(hidden_dim + num_dim, output_dim).to(DEVICE)
        def forward(xb_text, xb_num):
            xs = _prepare_sequence_from_embeddings(xb_text)
            h = base(xs)
            z = torch.cat([h, xb_num], dim=1)
            return head(z)
        model = torch.nn.Module(); model.base = base; model.head = head; model.to(DEVICE)
    elif model_name == 'Transformer':
        seq = _prepare_sequence_from_embeddings(X_text)
        base = TransformerEncoderModel(input_dim=seq.shape[2], num_heads=4, num_layers=2, hidden_dim=256, output_dim=hidden_dim, dropout=0.2).to(DEVICE)
        head = torch.nn.Linear(hidden_dim + num_dim, output_dim).to(DEVICE)
        def forward(xb_text, xb_num):
            xs = _prepare_sequence_from_embeddings(xb_text)
            h = base(xs)
            z = torch.cat([h, xb_num], dim=1)
            return head(z)
        model = torch.nn.Module(); model.base = base; model.head = head; model.to(DEVICE)
    else:
        return None, {}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 256
    n = X_text.shape[0]
    # simple train/val split
    split = int(0.8 * n)
    indices = torch.randperm(n, device=DEVICE)
    train_idx, val_idx = indices[:split], indices[split:]

    def run_epoch(idxs, train=True):
        if train:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        all_true, all_pred = [], []
        for start in range(0, len(idxs), batch_size):
            batch_ids = idxs[start:start+batch_size]
            xb_text = X_text[batch_ids]
            xb_num = X_num[batch_ids]
            yb = y[batch_ids]
            with torch.set_grad_enabled(train):
                out = forward(xb_text, xb_num)
                if task_type == 'classification':
                    loss = criterion(out, yb)
                    preds = out.argmax(dim=1).detach().cpu().numpy()
                    all_pred.extend(list(preds))
                    all_true.extend(list(yb.detach().cpu().numpy()))
                else:
                    out = out.squeeze(-1)
                    loss = criterion(out, yb)
                    all_pred.extend(list(out.detach().cpu().numpy()))
                    all_true.extend(list(yb.detach().cpu().numpy()))
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += float(loss.detach().cpu())
        metrics = {}
        if task_type == 'classification':
            metrics = classification_metrics(all_true, all_pred)
        else:
            from utils.metrics import regression_metrics
            metrics = regression_metrics(all_true, all_pred)
        return total_loss / max(1, (len(idxs) // batch_size)), metrics

    train_hist, val_hist = [], []
    for _ in range(epochs):
        tr_loss, tr_metrics = run_epoch(train_idx, train=True)
        _, va_metrics = run_epoch(val_idx, train=False)
        train_hist.append(tr_metrics)
        val_hist.append(va_metrics)
    return model, {"train": train_hist[-1], "val": val_hist[-1]}

def main():
    st.title("✨ Multi-Modal Amazon Review Analyzer")

    # Sidebar file selection
    st.sidebar.header("Data Files")
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
    metadata_choice = st.sidebar.selectbox(
        "Select metadata JSONL",
        options=raw_files,
        index=raw_files.index(default_metadata) if default_metadata in raw_files else 0,
        key="metadata_file_choice",
    ) if raw_files else None

    selected_reviews_path = os.path.join(RAW_DATA_DIR, reviews_choice) if reviews_choice else REVIEWS_FILE
    selected_metadata_path = os.path.join(RAW_DATA_DIR, metadata_choice) if metadata_choice else METADATA_FILE

    # Load processed data directly for fast UI
    with st.spinner("Loading processed data..."):
        parquet_path = ensure_processed_parquet(selected_reviews_path)
        reviews_df = load_processed_data(parquet_path, sample_size=100000)
        
        # Load metadata separately if needed
        metadata_df = load_raw_data(selected_metadata_path, selected_metadata_path)

    # Show basic stats
    st.sidebar.header("Dataset Info")
    st.sidebar.write(f"Total Reviews: {len(reviews_df)}")
    st.sidebar.write(f"Total Products in Metadata: {len(metadata_df)}")

    # Show rating distribution chart
    if "overall" in reviews_df.columns:
        plot_rating_distribution(reviews_df, rating_column="overall", key="sidebar_rating_dist")

    tabs = st.tabs(["Overview", "Explore", "Train", "Inference", "Advanced Analysis"])

    # Overview
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='metric'><span class='section-title'>Reviews</span><h2>" + str(len(reviews_df)) + "</h2></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='metric'><span class='section-title'>Products</span><h2>" + str(len(metadata_df)) + "</h2></div>", unsafe_allow_html=True)
        with c3:
            emb_cols = get_embedding_columns(reviews_df)
            st.markdown("<div class='metric'><span class='section-title'>Embedding Dim</span><h2>" + (str(len(emb_cols)) if emb_cols else "0") + "</h2></div>", unsafe_allow_html=True)

        st.markdown("<div class='vercel-card'>", unsafe_allow_html=True)
        st.subheader("Rating Distribution")
        if "overall" in reviews_df.columns:
            plot_rating_distribution(reviews_df, rating_column="overall", key="overview_rating_dist")
        else:
            st.info("Column 'overall' not found in dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Explore
    with tabs[1]:
        st.markdown("<div class='vercel-card'>", unsafe_allow_html=True)
        st.subheader("Sample Review Preprocessing")
        preprocessor = get_preprocessor()
        example_text = reviews_df["reviewText"].iloc[0] if "reviewText" in reviews_df and len(reviews_df) else ""
        cleaned_text = preprocessor.clean_text(example_text)
        filtered_text = preprocessor.remove_stopwords(cleaned_text)
        st.write({
            "original": example_text,
            "cleaned": cleaned_text,
            "without_stopwords": filtered_text
        })
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='vercel-card'>", unsafe_allow_html=True)
        st.subheader("SBERT Embedding Preview")
        embedding = None
        try:
            encoder = get_encoder()
            embedding = encoder.encode([filtered_text], max_length=MAX_SEQ_LENGTH)
        except Exception as exc:
            st.warning(f"Could not compute SBERT embedding: {exc}")
        if embedding is not None:
            st.write({"first_10_dims": embedding[0][:10].tolist()})
        st.markdown("</div>", unsafe_allow_html=True)

    # Train
    with tabs[2]:
        st.markdown("<div class='vercel-card'>", unsafe_allow_html=True)
        st.subheader("Model Training")
        c1, c2, c3 = st.columns(3)
        with c1:
            task_type = st.selectbox("Task", options=["classification", "regression"], index=0)
        with c2:
            model_choice = st.selectbox("Model", options=["EarlyFusion", "HybridFusion", "LSTM", "Transformer"], index=0)
        with c3:
            epochs = st.slider("Epochs", 1, 10, 3)
        go = st.button("Train / Evaluate")
        if go:
            with st.spinner("Training model..."):
                model, metrics = train_model(reviews_df, model_choice, task_type, epochs=epochs)
            if model is None:
                st.error("Embeddings not found in data. Make sure processing ran correctly.")
            else:
                st.success("Training complete.")
                st.write("Validation metrics:")
                st.json(metrics.get("val", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    # Inference
    with tabs[3]:
        st.markdown("<div class='vercel-card'>", unsafe_allow_html=True)
        st.subheader("Inference Playground")
        user_text = st.text_area("Enter a review to analyze", "This is a fantastic product! Highly recommended.")
        if st.button("Embed and show features", key="infer_btn"):
            preprocessor = get_preprocessor()
            ct = preprocessor.clean_text(user_text)
            ct = preprocessor.remove_stopwords(ct)
            try:
                vec = get_encoder().encode([ct], max_length=MAX_SEQ_LENGTH)[0]
                st.write("First 10 embedding dims:", vec[:10].tolist())
            except Exception as exc:
                st.warning(f"Could not compute embedding: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Advanced Analysis Tab
    with tabs[4]:
        st.markdown('<div class="vercel-card">', unsafe_allow_html=True)
        st.header("🔬 Advanced Multimodal Analysis")
        
        # Model Performance Comparison
        st.subheader("📊 Model Performance Comparison")
        
        # Mock results for demonstration (in real implementation, these would come from actual training)
        mock_results = {
            "EarlyFusion": {"accuracy": 0.85, "f1_score": 0.83, "precision": 0.84, "recall": 0.82},
            "LateFusion": {"accuracy": 0.87, "f1_score": 0.85, "precision": 0.86, "recall": 0.84},
            "HybridFusion": {"accuracy": 0.89, "f1_score": 0.87, "precision": 0.88, "recall": 0.86},
            "CrossModalTransformer": {"accuracy": 0.91, "f1_score": 0.89, "precision": 0.90, "recall": 0.88},
            "LSTM": {"accuracy": 0.82, "f1_score": 0.80, "precision": 0.81, "recall": 0.79},
            "Transformer": {"accuracy": 0.84, "f1_score": 0.82, "precision": 0.83, "recall": 0.81}
        }
        
        # Model comparison chart
        comparison_fig = create_model_comparison_chart(mock_results, "classification")
        st.plotly_chart(comparison_fig, use_container_width=True, key="model_comparison_chart")
        
        # Fusion technique comparison
        st.subheader("🔄 Fusion Technique Analysis")
        fusion_fig = create_fusion_technique_comparison({
            "classification": mock_results,
            "regression": {k: {"mae": 0.5, "rmse": 0.7} for k in mock_results.keys()}
        })
        st.plotly_chart(fusion_fig, use_container_width=True, key="fusion_technique_chart")
        
        # Cross-modal feature analysis
        st.subheader("🎯 Cross-Modal Feature Analysis")
        
        # Mock feature data
        sample_text_features = {
            "polarity": 0.6, "subjectivity": 0.4, "word_count": 45, 
            "sentiment_intensity": 0.8, "sentiment_confidence": 0.9
        }
        sample_numerical_features = {
            "rating": 4.5, "helpful_votes": 12, "review_count": 1
        }
        
        cross_modal_fig = create_cross_modal_analysis_chart(sample_text_features, sample_numerical_features)
        st.plotly_chart(cross_modal_fig, use_container_width=True, key="cross_modal_analysis_chart")
        
        # Performance radar chart
        st.subheader("🎯 Performance Radar Chart")
        radar_fig = create_performance_radar_chart(mock_results)
        st.plotly_chart(radar_fig, use_container_width=True, key="performance_radar_chart")
        
        # Advanced metrics analysis
        st.subheader("📈 Advanced Metrics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Ranking:**")
            ranking = advanced_model_comparison(mock_results)
            for i, (model, score) in enumerate(ranking["overall_ranking"], 1):
                st.write(f"{i}. {model}: {score:.3f}")
        
        with col2:
            st.write("**Best Model by Metric:**")
            for metric, data in ranking.items():
                if metric != "overall_ranking" and metric != "model_scores":
                    st.write(f"**{metric.replace('_', ' ').title()}**: {data['best_model']} ({data['best_value']:.3f})")
        
        # Generate and display model report
        st.subheader("📋 Comprehensive Model Report")
        
        if st.button("Generate Model Report"):
            report = generate_model_report(mock_results, "classification")
            st.markdown(report)
            
            # Export to HTML
            html_filename = export_results_to_html(mock_results, "model_analysis_report.html")
            st.success(f"Report exported to {html_filename}")
            
            # Provide download link
            with open(html_filename, 'rb') as f:
                st.download_button(
                    label="📥 Download Interactive Report",
                    data=f.read(),
                    file_name="multimodal_analysis_report.html",
                    mime="text/html"
                )
        
        # Feature engineering demonstration
        st.subheader("🔧 Cross-Modal Feature Engineering Demo")
        
        demo_text = st.text_area("Enter text for feature analysis:", 
                                "This product is absolutely amazing! The quality is outstanding and I highly recommend it.")
        
        if st.button("Analyze Features"):
            # Extract advanced sentiment features
            sentiment_features = advanced_sentiment_features(demo_text)
            linguistic_features = extract_linguistic_features(demo_text)
            
            # Mock numerical features
            mock_num_features = {"rating": 4.5, "helpful_votes": 8, "review_count": 1}
            
            # Create cross-modal features
            cross_modal_features = cross_modal_feature_engineering(sentiment_features, mock_num_features)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Sentiment Features:**")
                for key, value in sentiment_features.items():
                    st.write(f"- {key}: {value:.3f}")
            
            with col2:
                st.write("**Linguistic Features:**")
                for key, value in linguistic_features.items():
                    st.write(f"- {key}: {value:.3f}")
            
            with col3:
                st.write("**Cross-Modal Features:**")
                for key, value in cross_modal_features.items():
                    st.write(f"- {key}: {value:.3f}")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
