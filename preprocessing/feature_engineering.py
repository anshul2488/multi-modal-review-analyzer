from textblob import TextBlob
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import re

def sentiment_features(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception:
        polarity, subjectivity = 0.0, 0.0
    return {"polarity": polarity, "subjectivity": subjectivity}

def advanced_sentiment_features(text: str) -> Dict[str, float]:
    """Enhanced sentiment analysis with multiple indicators"""
    try:
        blob = TextBlob(text)
        
        # Basic sentiment
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Text complexity features
        word_count = len(text.split())
        char_count = len(text)
        avg_word_length = char_count / max(word_count, 1)
        
        # Emotional indicators
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
        
        # Sentiment intensity
        sentiment_intensity = abs(polarity)
        
        # Confidence based on subjectivity
        sentiment_confidence = 1 - subjectivity
        
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "caps_ratio": caps_ratio,
            "sentiment_intensity": sentiment_intensity,
            "sentiment_confidence": sentiment_confidence
        }
    except Exception:
        return {
            "polarity": 0.0, "subjectivity": 0.0, "word_count": 0, "avg_word_length": 0.0,
            "exclamation_count": 0, "question_count": 0, "caps_ratio": 0.0,
            "sentiment_intensity": 0.0, "sentiment_confidence": 0.0
        }

def cross_modal_feature_engineering(text_features: Dict, numerical_features: Dict) -> Dict[str, float]:
    """Create cross-modal features that combine text and numerical information"""
    
    cross_modal_features = {}
    
    # Feature interaction terms
    if 'polarity' in text_features and 'rating' in numerical_features:
        cross_modal_features['sentiment_rating_alignment'] = text_features['polarity'] * numerical_features['rating']
        cross_modal_features['sentiment_rating_discrepancy'] = abs(text_features['polarity'] - (numerical_features['rating'] - 3) / 2)
    
    # Confidence-weighted features
    if 'sentiment_confidence' in text_features:
        cross_modal_features['confidence_weighted_polarity'] = text_features['polarity'] * text_features['sentiment_confidence']
    
    # Length-based interactions
    if 'word_count' in text_features and 'review_count' in numerical_features:
        cross_modal_features['word_count_normalized'] = text_features['word_count'] / max(numerical_features['review_count'], 1)
    
    # Emotional intensity interactions
    if 'sentiment_intensity' in text_features and 'helpful_votes' in numerical_features:
        cross_modal_features['intensity_helpfulness_correlation'] = text_features['sentiment_intensity'] * numerical_features['helpful_votes']
    
    return cross_modal_features

def compute_cross_modal_attention_weights(text_emb: torch.Tensor, num_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute attention weights for cross-modal fusion"""
    
    # Simple dot-product attention
    attention_scores = torch.matmul(text_emb, num_emb.T)
    
    # Normalize attention weights
    text_attention = F.softmax(attention_scores.mean(dim=1), dim=0)
    num_attention = F.softmax(attention_scores.mean(dim=0), dim=0)
    
    return text_attention, num_attention

def create_fusion_features(text_emb: torch.Tensor, num_features: torch.Tensor) -> torch.Tensor:
    """Create advanced fusion features using multiple strategies"""
    
    # 1. Element-wise multiplication (Hadamard product)
    hadamard = text_emb * num_features
    
    # 2. Concatenation
    concatenated = torch.cat([text_emb, num_features], dim=1)
    
    # 3. Addition
    added = text_emb + num_features
    
    # 4. Cross-attention weighted features
    text_att, num_att = compute_cross_modal_attention_weights(text_emb, num_features)
    text_weighted = text_emb * text_att.unsqueeze(-1)
    num_weighted = num_features * num_att.unsqueeze(-1)
    
    # Combine all fusion strategies
    fusion_features = torch.cat([
        hadamard,
        concatenated,
        added,
        text_weighted,
        num_weighted
    ], dim=1)
    
    return fusion_features

def extract_linguistic_features(text: str) -> Dict[str, float]:
    """Extract linguistic features from text"""
    
    # Word-level features
    words = text.split()
    word_count = len(words)
    
    # Character-level features
    char_count = len(text)
    avg_word_length = char_count / max(word_count, 1)
    
    # Punctuation features
    punctuation_count = sum(1 for c in text if c in '.,!?;:')
    punctuation_ratio = punctuation_count / max(char_count, 1)
    
    # Case features
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_count / max(char_count, 1)
    
    # Repetition features
    unique_words = len(set(words))
    lexical_diversity = unique_words / max(word_count, 1)
    
    # Emotional markers
    positive_words = len(re.findall(r'\b(good|great|excellent|amazing|wonderful|fantastic|love|like|best)\b', text.lower()))
    negative_words = len(re.findall(r'\b(bad|terrible|awful|hate|worst|horrible|disappointing)\b', text.lower()))
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'punctuation_ratio': punctuation_ratio,
        'uppercase_ratio': uppercase_ratio,
        'lexical_diversity': lexical_diversity,
        'positive_word_count': positive_words,
        'negative_word_count': negative_words,
        'sentiment_lexicon_score': (positive_words - negative_words) / max(word_count, 1)
    }

def aggregate_numerical_features(df):
    # df = DataFrame with numeric columns to aggregate (for example: 'rating', 'helpful_votes', etc.)
    agg_dict = {
        "review_count": df.shape[0],
        "avg_rating": np.mean(df["rating"]) if "rating" in df else 0,
        "total_helpful_votes": np.sum(df["helpful_votes"]) if "helpful_votes" in df else 0,
        "avg_review_length": np.mean(df["review_text"].apply(lambda x: len(x.split()))) if "review_text" in df else 0,
    }
    return agg_dict
