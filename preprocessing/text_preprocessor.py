"""
Text Preprocessing Module for Review Analysis.

This module provides text preprocessing functionality including cleaning,
stopword removal, and basic feature extraction for the multimodal review analyzer.
"""

import re
import nltk
from nltk.corpus import stopwords


class TextPreprocessor:
    """
    Text preprocessor for cleaning and processing review text.
    
    This class provides methods for text cleaning, stopword removal, and
    basic feature extraction from review texts.
    
    Args:
        language (str): Language for stopword removal (default: 'english')
    """
    
    def __init__(self, language='english'):
        """
        Initialize text preprocessor with stopword list.
        
        Args:
            language (str): Language identifier for stopword corpus
        """
        try:
            # Load stopwords for the specified language
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            # Download stopwords corpus if not available
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words(language))

    def clean_text(self, text):
        """
        Clean and normalize input text.
        
        This method performs the following cleaning operations:
        - Remove HTML tags
        - Remove special characters (keep alphanumeric and whitespace)
        - Convert to lowercase
        - Normalize whitespace
        
        Args:
            text (str): Raw input text to clean
            
        Returns:
            str: Cleaned and normalized text
        """
        # Handle non-string inputs
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags using regex
        text = re.sub('<.*?>', '', text)
        
        # Remove special characters, keeping only alphanumeric and whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Normalize whitespace: split and rejoin to remove extra spaces
        text = ' '.join(text.split())
        
        return text

    def remove_stopwords(self, text):
        """
        Remove stopwords from text.
        
        Args:
            text (str): Input text with potential stopwords
            
        Returns:
            str: Text with stopwords removed
        """
        # Tokenize text by splitting on whitespace
        tokens = text.split()
        
        # Filter out stopwords
        filtered = [word for word in tokens if word not in self.stop_words]
        
        # Rejoin filtered tokens into text string
        return ' '.join(filtered)

    def basic_features(self, text):
        """
        Extract basic statistical features from text.
        
        Computes word count, character count, and average word length.
        
        Args:
            text (str): Input text for feature extraction
            
        Returns:
            dict: Dictionary containing:
                - word_count: Number of words in text
                - char_count: Number of characters in text
                - avg_word_length: Average length of words
        """
        # Tokenize text into words
        words = text.split()
        
        # Compute basic statistics
        n_words = len(words)
        n_chars = len(text)
        
        # Calculate average word length (handle division by zero)
        avg_word_len = sum(len(word) for word in words) / n_words if n_words else 0
        
        return {
            "word_count": n_words,
            "char_count": n_chars,
            "avg_word_length": avg_word_len
        }
