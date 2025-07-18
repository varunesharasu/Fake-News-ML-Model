import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import re

class FeatureEngineer:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.svd = None
        
    def extract_text_features(self, df):
        """Extract various text-based features"""
        features_df = df.copy()
        
        # Basic text statistics
        features_df['char_count'] = df['cleaned_content'].apply(len)
        features_df['word_count'] = df['cleaned_content'].apply(lambda x: len(x.split()))
        features_df['sentence_count'] = df['cleaned_content'].apply(lambda x: len(x.split('.')))
        features_df['avg_word_length'] = df['cleaned_content'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        
        # Uppercase words ratio
        features_df['uppercase_ratio'] = df['content'].apply(
            lambda x: len(re.findall(r'[A-Z]', str(x))) / len(str(x)) if len(str(x)) > 0 else 0
        )
        
        # Punctuation count
        features_df['punctuation_count'] = df['content'].apply(
            lambda x: len(re.findall(r'[!?.,;:]', str(x)))
        )
        
        # Exclamation marks
        features_df['exclamation_count'] = df['content'].apply(
            lambda x: str(x).count('!')
        )
        
        # Question marks
        features_df['question_count'] = df['content'].apply(
            lambda x: str(x).count('?')
        )
        
        return features_df
    
    def create_tfidf_features(self, train_texts, test_texts=None, max_features=5000):
        """Create TF-IDF features"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.8
        )
        
        # Fit on training data
        tfidf_train = self.tfidf_vectorizer.fit_transform(train_texts)
        
        if test_texts is not None:
            tfidf_test = self.tfidf_vectorizer.transform(test_texts)
            return tfidf_train, tfidf_test
        
        return tfidf_train
    
    def create_count_features(self, train_texts, test_texts=None, max_features=5000):
        """Create Count Vectorizer features"""
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.8
        )
        
        # Fit on training data
        count_train = self.count_vectorizer.fit_transform(train_texts)
        
        if test_texts is not None:
            count_test = self.count_vectorizer.transform(test_texts)
            return count_train, count_test
        
        return count_train
    
    def apply_dimensionality_reduction(self, X_train, X_test=None, n_components=100):
        """Apply SVD for dimensionality reduction"""
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        X_train_reduced = self.svd.fit_transform(X_train)
        
        if X_test is not None:
            X_test_reduced = self.svd.transform(X_test)
            return X_train_reduced, X_test_reduced
        
        return X_train_reduced