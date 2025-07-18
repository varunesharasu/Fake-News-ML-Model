import pandas as pd
import numpy as np
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import contractions
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def load_data(self, fake_path, true_path):
        """Load and combine fake and true news datasets"""
        try:
            fake_df = pd.read_csv(fake_path)
            true_df = pd.read_csv(true_path)
            
            # Add labels
            fake_df['label'] = 1  # 1 for fake
            true_df['label'] = 0  # 0 for real
            
            # Combine datasets
            combined_df = pd.concat([fake_df, true_df], ignore_index=True)
            
            print(f"Fake news articles: {len(fake_df)}")
            print(f"True news articles: {len(true_df)}")
            print(f"Total articles: {len(combined_df)}")
            
            return combined_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Expand contractions
        text = contractions.fix(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def lemmatize_text(self, text):
        """Lemmatize text using spaCy"""
        doc = self.nlp(text)
        lemmatized = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(lemmatized)
    
    def get_sentiment_features(self, text):
        """Extract sentiment features"""
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        return {
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu']
        }
    
    def preprocess_dataframe(self, df):
        """Preprocess the entire dataframe"""
        processed_df = df.copy()
        
        # Combine title and text if both exist
        if 'title' in df.columns and 'text' in df.columns:
            processed_df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        elif 'title' in df.columns:
            processed_df['content'] = df['title'].fillna('')
        elif 'text' in df.columns:
            processed_df['content'] = df['text'].fillna('')
        else:
            print("No title or text column found!")
            return None
        
        # Clean text
        print("Cleaning text...")
        processed_df['cleaned_content'] = processed_df['content'].apply(self.clean_text)
        
        # Remove stopwords
        print("Removing stopwords...")
        processed_df['content_no_stopwords'] = processed_df['cleaned_content'].apply(self.remove_stopwords)
        
        # Lemmatize
        print("Lemmatizing...")
        processed_df['lemmatized_content'] = processed_df['cleaned_content'].apply(self.lemmatize_text)
        
        # Extract sentiment features
        print("Extracting sentiment features...")
        sentiment_features = processed_df['cleaned_content'].apply(self.get_sentiment_features)
        sentiment_df = pd.json_normalize(sentiment_features)
        
        # Combine with main dataframe
        processed_df = pd.concat([processed_df, sentiment_df], axis=1)
        
        return processed_df