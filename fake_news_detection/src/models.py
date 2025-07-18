import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

class TraditionalModels:
    def __init__(self):
        self.models = {}
        
    def initialize_models(self):
        """Initialize all traditional ML models"""
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': MultinomialNB(),
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'lightgbm': LGBMClassifier(random_state=42, verbose=-1),
            'catboost': CatBoostClassifier(random_state=42, verbose=False)
        }
        
    def train_individual_models(self, X_train, y_train):
        """Train all individual models"""
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
        return trained_models
    
    def create_ensemble(self, X_train, y_train):
        """Create ensemble model"""
        # Select best performing models for ensemble
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
            ('lgb', LGBMClassifier(random_state=42, verbose=-1))
        ]
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        return ensemble

class DeepLearningModels:
    def __init__(self):
        self.tokenizer = None
        self.max_length = 512
        
    def create_cnn_model(self, vocab_size, embedding_dim=100, max_length=512):
        """Create CNN model for text classification"""
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_lstm_model(self, vocab_size, embedding_dim=100, max_length=512):
        """Create LSTM model for text classification"""
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_sequences(self, texts, vocab_size=10000, max_length=512):
        """Prepare text sequences for deep learning models"""
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
        
        return padded_sequences
    
    def create_bert_model(self, model_name='bert-base-uncased', num_labels=2):
        """Create BERT model for classification"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        return tokenizer, model

class HybridModel:
    def __init__(self):
        self.traditional_models = {}
        self.deep_models = {}
        self.meta_model = None
        
    def train_hybrid_model(self, X_traditional, X_deep, y_train):
        """Train hybrid model combining traditional and deep learning"""
        # Train traditional models
        traditional_ml = TraditionalModels()
        traditional_ml.initialize_models()
        self.traditional_models = traditional_ml.train_individual_models(X_traditional, y_train)
        
        # Get predictions from traditional models
        traditional_preds = []
        for name, model in self.traditional_models.items():
            pred_proba = model.predict_proba(X_traditional)[:, 1]
            traditional_preds.append(pred_proba)
        
        traditional_preds = np.column_stack(traditional_preds)
        
        # Train meta-model
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(traditional_preds, y_train)
        
        return self