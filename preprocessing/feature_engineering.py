"""
Advanced Feature Engineering Module for Cross-Modal Analysis.

This module provides comprehensive feature extraction capabilities including
sentiment analysis, linguistic features, cross-modal feature engineering,
and fusion feature creation for the multimodal review analyzer.
"""

from textblob import TextBlob
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import re


def sentiment_features(text):
    """
    Extract basic sentiment features from text using TextBlob.
    
    Computes polarity (sentiment strength, range -1 to 1) and subjectivity
    (opinion vs fact, range 0 to 1).
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Dictionary containing 'polarity' and 'subjectivity' scores
    """
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception:
        polarity, subjectivity = 0.0, 0.0
    return {"polarity": polarity, "subjectivity": subjectivity}

def advanced_sentiment_features(text: str) -> Dict[str, float]:
    """
    Extract advanced sentiment features with multiple indicators.
    
    Computes comprehensive sentiment analysis including basic sentiment scores,
    text complexity metrics, emotional indicators, and confidence measures.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Dictionary containing multiple sentiment and text features:
            - polarity: Sentiment polarity (-1 to 1)
            - subjectivity: Subjectivity score (0 to 1)
            - word_count: Number of words
            - avg_word_length: Average word length
            - exclamation_count: Number of exclamation marks
            - question_count: Number of question marks
            - caps_ratio: Ratio of uppercase characters
            - sentiment_intensity: Absolute polarity value
            - sentiment_confidence: Confidence based on subjectivity
    """
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
    """
    Create cross-modal features that combine text and numerical information.
    
    Generates interaction features that capture relationships between text sentiment
    and numerical ratings, helpfulness metrics, and review counts.
    
    Args:
        text_features (dict): Text-based features (sentiment, linguistic features)
        numerical_features (dict): Numerical features (ratings, votes, counts)
        
    Returns:
        dict: Cross-modal interaction features including:
            - sentiment_rating_alignment: Product of sentiment and rating
            - sentiment_rating_discrepancy: Absolute difference between sentiment and normalized rating
            - confidence_weighted_polarity: Polarity weighted by confidence
            - word_count_normalized: Word count normalized by review count
            - intensity_helpfulness_correlation: Sentiment intensity correlated with helpful votes
    """
    
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
    """
    Compute attention weights for cross-modal fusion using dot-product attention.
    
    Calculates attention scores between text and numerical embeddings to determine
    the importance of each modality for fusion.
    
    Args:
        text_emb (torch.Tensor): Text embeddings tensor
        num_emb (torch.Tensor): Numerical embeddings tensor
        
    Returns:
        tuple: (text_attention_weights, num_attention_weights) - Softmax-normalized attention weights
    """
    
    # Simple dot-product attention
    attention_scores = torch.matmul(text_emb, num_emb.T)
    
    # Normalize attention weights
    text_attention = F.softmax(attention_scores.mean(dim=1), dim=0)
    num_attention = F.softmax(attention_scores.mean(dim=0), dim=0)
    
    return text_attention, num_attention

def create_fusion_features(text_emb: torch.Tensor, num_features: torch.Tensor) -> torch.Tensor:
    """
    Create advanced fusion features using multiple fusion strategies.
    
    Combines text and numerical embeddings using various fusion techniques:
    Hadamard product (element-wise multiplication), concatenation, addition,
    and attention-weighted features.
    
    Args:
        text_emb (torch.Tensor): Text embedding tensor
        num_features (torch.Tensor): Numerical feature tensor
        
    Returns:
        torch.Tensor: Concatenated fusion features from all strategies
    """
    
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
    """
    Extract comprehensive linguistic features from text.
    
    Computes word-level, character-level, punctuation, case, and emotional
    marker features to capture linguistic patterns in the text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Dictionary containing linguistic features:
            - word_count: Number of words
            - char_count: Number of characters
            - avg_word_length: Average word length
            - punctuation_ratio: Ratio of punctuation to total characters
            - uppercase_ratio: Ratio of uppercase to total characters
            - lexical_diversity: Unique words / total words ratio
            - positive_word_count: Count of positive sentiment words
            - negative_word_count: Count of negative sentiment words
            - sentiment_lexicon_score: Normalized positive-negative word difference
    """
    
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
    """
    Aggregate numerical features from a DataFrame.
    
    Computes summary statistics from numerical columns including review count,
    average rating, total helpful votes, and average review length.
    
    Args:
        df (pandas.DataFrame): DataFrame containing review data with numerical columns
        
    Returns:
        dict: Dictionary containing aggregated numerical features:
            - review_count: Total number of reviews
            - avg_rating: Average rating value
            - total_helpful_votes: Sum of helpful votes
            - avg_review_length: Average number of words per review
    """
    agg_dict = {
        "review_count": df.shape[0],
        "avg_rating": np.mean(df["rating"]) if "rating" in df else 0,
        "total_helpful_votes": np.sum(df["helpful_votes"]) if "helpful_votes" in df else 0,
        "avg_review_length": np.mean(df["review_text"].apply(lambda x: len(x.split()))) if "review_text" in df else 0,
    }
    return agg_dict
