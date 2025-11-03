# Multimodal Review Analyzer

A comprehensive system for analyzing product reviews using advanced multimodal fusion techniques, featuring Early Fusion, Late Fusion, Hybrid Fusion, and Cross-Modal Transformer architectures with interactive deployment capabilities.

## Features

### Advanced Fusion Techniques
- **Early Fusion**: Concatenates text and numerical features before processing with residual connections
- **Late Fusion**: Processes modalities separately with decision-level fusion and attention mechanisms
- **Hybrid Fusion**: Combines early and late fusion with bidirectional attention and gating mechanisms
- **Cross-Modal Transformer**: Uses transformer architecture with cross-modal attention for enhanced feature interaction
- **Ensemble Models**: Advanced ensemble strategies with adaptive weighting and confidence estimation

### Deep Learning Models
- **LSTM**: Long Short-Term Memory for sequential text processing
- **Transformer**: Self-attention based text processing with positional encoding
- **Custom Fusion Architectures**: Novel cross-modal fusion implementations

### Interactive Dashboard
- **Modern UI**: Vercel-inspired design with responsive layout
- **Real-time Analysis**: Live model training and inference capabilities
- **Advanced Visualizations**: Performance comparison charts, radar plots, and cross-modal analysis
- **Export Functionality**: Interactive HTML reports with downloadable analysis

### Cross-Modal Analysis
- **Feature Engineering**: Advanced sentiment analysis, linguistic features, and cross-modal interactions
- **Attention Mechanisms**: Cross-modal attention weights and fusion effectiveness analysis
- **Novel Metrics**: Cross-modal consistency and fusion effectiveness evaluation

## Project Structure

```
multi_modal_review_analyzer/
├── app.py                          # Main Streamlit application with interactive dashboard
├── config.py                       # Configuration settings
├── train_models.py                 # Enhanced model training with all fusion techniques
├── models/                         # Model implementations
│   ├── fusion_model.py               # Advanced fusion architectures + ensemble models
│   ├── lstm_model.py                 # LSTM implementation
│   ├── transformer_model.py          # Transformer implementation
│   └── nlp_utils.py                  # SBERT and NLP utilities
├── preprocessing/                  # Data preprocessing and feature engineering
│   ├── text_preprocessor.py          # Text preprocessing pipeline
│   └── feature_engineering.py        # Cross-modal feature engineering
├── utils/                         # Utility functions
│   ├── data_loader.py                # Data loading and preprocessing
│   ├── metrics.py                    # Advanced evaluation metrics
│   └── visualization.py              # Interactive visualization tools
├── data/                          # Data directory
│   ├── raw/                          # Raw JSONL data files
│   └── processed/                    # Processed Parquet files
└── reports/                       # Generated analysis reports
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd multi_modal_review_analyzer

# Create and activate virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Interactive Application

```bash
# Start the Streamlit dashboard
streamlit run app.py
```

**Access the application at**: `http://localhost:8501`

### 3. Training Models

```bash
# Train all fusion models
python train_models.py

# Train specific models
python train_models.py --models EarlyFusion LateFusion HybridFusion CrossModalTransformer
```

## Model Architectures

### Fusion Techniques

#### 1. Early Fusion Model
- **Architecture**: Concatenates text and numerical features before processing
- **Enhancements**: Layer normalization, residual connections, sequential feature transformation
- **Use Case**: When modalities are highly correlated

#### 2. Late Fusion Model
- **Architecture**: Separate processing branches with decision-level fusion
- **Features**: Attention-based fusion mechanism, modality-specific feature extraction
- **Use Case**: When modalities have different optimal processing strategies

#### 3. Hybrid Fusion Model
- **Architecture**: Combines early and late fusion with advanced attention mechanisms
- **Features**: Bidirectional cross-modal attention, gating mechanisms, self-attention
- **Use Case**: Optimal balance between early and late fusion benefits

#### 4. Cross-Modal Transformer
- **Architecture**: Transformer-based cross-modal attention
- **Features**: Learned positional encoding, multi-layer transformer blocks
- **Use Case**: Complex cross-modal interactions and long-range dependencies

#### 5. Ensemble Models
- **Standard Ensemble**: Weighted combination of all fusion strategies
- **Adaptive Ensemble**: Learns when to use which fusion strategy with confidence estimation
- **Use Case**: Maximum performance through intelligent model combination

### Individual Models

#### LSTM Model
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Features**: Sequential processing, attention weights, dropout regularization

#### Transformer Model
- **Architecture**: Multi-head self-attention with positional encoding
- **Features**: Layer normalization, feed-forward networks, residual connections

## Advanced Features

### Cross-Modal Feature Engineering

```python
# Advanced sentiment analysis
sentiment_features = advanced_sentiment_features(text)

# Linguistic feature extraction
linguistic_features = extract_linguistic_features(text)

# Cross-modal feature interactions
cross_modal_features = cross_modal_feature_engineering(sentiment_features, numerical_features)

# Fusion feature creation
fusion_features = create_fusion_features(text_embeddings, numerical_features)
```

### Interactive Dashboard Features

1. **Overview Tab**: Data statistics and distribution analysis
2. **Explore Tab**: Interactive data exploration with filtering
3. **Train Tab**: Model training with real-time progress monitoring
4. **Inference Tab**: Live prediction with confidence scores
5. **Advanced Analysis Tab**: 
   - Model performance comparison charts
   - Fusion technique analysis
   - Cross-modal feature importance
   - Performance radar charts
   - Interactive HTML report generation

### Evaluation Metrics

#### Classification Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **F1-Macro/Micro**: Different averaging strategies

#### Regression Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R² Score**: Coefficient of determination
- **Explained Variance**: Proportion of variance explained

#### Cross-Modal Metrics
- **Cross-Modal Consistency**: Agreement between modalities
- **Fusion Effectiveness**: Improvement from fusion strategies
- **Feature Importance**: Cross-modal feature contribution analysis

## Configuration

Key parameters in `config.py`:

```python
# Model Configuration
MAX_SEQ_LENGTH = 512          # Maximum text sequence length
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"  # SBERT model for embeddings
BATCH_SIZE = 32               # Training batch size
LEARNING_RATE = 0.001         # Learning rate
NUM_EPOCHS = 50               # Training epochs

# Fusion Configuration
HIDDEN_DIM = 128              # Hidden layer dimensions
DROPOUT_RATE = 0.2            # Dropout probability
ATTENTION_HEADS = 8           # Number of attention heads
```

## Performance Optimization

### Model Optimization Features
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Model Checkpointing**: Saves best models during training

### Ensemble Strategies
- **Weighted Ensemble**: Learnable weights for model combination
- **Adaptive Ensemble**: Dynamic model selection based on input characteristics
- **Confidence Estimation**: Uncertainty quantification for predictions

## Data Format

### Input Data Structure (JSONL)
```json
{
  "reviewText": "This product is absolutely amazing! The quality is outstanding...",
  "overall": 5,
  "helpful": [12, 15],
  "asin": "B08N5WRWNW",
  "reviewerID": "A1D87F6ZCVE5NK",
  "reviewTime": "12 7, 2020"
}
```

### Processed Data Features
- **Text Features**: SBERT embeddings, sentiment scores, linguistic features
- **Numerical Features**: Ratings, helpful votes, review counts
- **Cross-Modal Features**: Interaction terms, confidence scores, alignment metrics

## Interactive HTML Reports

The system generates comprehensive HTML reports with:

- **Interactive Charts**: Plotly-powered visualizations
- **Model Comparisons**: Side-by-side performance analysis
- **Fusion Analysis**: Detailed fusion technique evaluation
- **Export Capabilities**: Downloadable reports and data

## Deployment

### Local Deployment
```bash
# Run with custom configuration
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Cloud Deployment
The application is ready for deployment on:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: With Procfile configuration
- **AWS/GCP**: Container-based deployment
- **Docker**: Containerized deployment

## Research Contributions

This project implements several novel contributions:

1. **Advanced Fusion Architectures**: Novel cross-modal attention mechanisms
2. **Ensemble Strategies**: Adaptive ensemble learning for multimodal data
3. **Cross-Modal Feature Engineering**: Advanced feature interaction techniques
4. **Interactive Analysis Tools**: Comprehensive visualization and analysis dashboard

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with proper documentation
4. Add tests for new functionality
5. Submit a pull request with detailed description

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **SBERT**: For semantic text embeddings
- **Streamlit**: For interactive web application framework
- **PyTorch**: For deep learning framework
- **Plotly**: For interactive visualizations

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation for detailed guides
