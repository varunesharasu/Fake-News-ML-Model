from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained model and preprocessors
try:
    model = joblib.load('models/saved_models/lightgbm_model.pkl')
    scaler = joblib.load('models/saved_models/scaler.pkl')
    feature_engineer = joblib.load('models/saved_models/feature_engineer.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    scaler = None
    feature_engineer = None

# Initialize data processor
data_processor = DataProcessor()

def predict_news(news_text):
    """Predict if news is fake or real"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Create a temporary dataframe
        temp_df = pd.DataFrame({
            'content': [news_text],
            'label': [0]  # Dummy label
        })
        
        # Preprocess the text
        temp_df['cleaned_content'] = temp_df['content'].apply(data_processor.clean_text)
        temp_df['content_no_stopwords'] = temp_df['cleaned_content'].apply(data_processor.remove_stopwords)
        temp_df['lemmatized_content'] = temp_df['content_no_stopwords']
        
        # Extract sentiment features
        sentiment_features = temp_df['cleaned_content'].apply(data_processor.get_sentiment_features)
        sentiment_df = pd.json_normalize(sentiment_features)
        temp_df = pd.concat([temp_df, sentiment_df], axis=1)
        
        # Extract text features
        features_df = feature_engineer.extract_text_features(temp_df)
        
        # Select the same features used in training
        feature_columns = ['char_count', 'word_count', 'sentence_count', 'avg_word_length',
                          'uppercase_ratio', 'punctuation_count', 'exclamation_count', 
                          'question_count', 'textblob_polarity', 'textblob_subjectivity',
                          'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral']
        
        X_features = features_df[feature_columns].fillna(0)
        
        # Create TF-IDF features
        tfidf_features = feature_engineer.tfidf_vectorizer.transform([temp_df['content_no_stopwords'].iloc[0]])
        
        # Scale the basic features
        X_features_scaled = scaler.transform(X_features)
        
        # Combine features
        X_combined = np.hstack([X_features_scaled, tfidf_features.toarray()])
        
        # Make prediction
        prediction = model.predict(X_combined)[0]
        prediction_proba = model.predict_proba(X_combined)[0]
        
        # Get confidence scores
        fake_confidence = prediction_proba[1] * 100
        real_confidence = prediction_proba[0] * 100
        
        result = {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'fake_confidence': round(fake_confidence, 2),
            'real_confidence': round(real_confidence, 2),
            'features': {
                'word_count': int(features_df['word_count'].iloc[0]),
                'char_count': int(features_df['char_count'].iloc[0]),
                'sentiment_polarity': round(features_df['textblob_polarity'].iloc[0], 3),
                'sentiment_subjectivity': round(features_df['textblob_subjectivity'].iloc[0], 3)
            }
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        news_text = data.get('news_text', '').strip()
        
        if not news_text:
            return jsonify({"error": "Please enter some news text"})
        
        if len(news_text) < 10:
            return jsonify({"error": "Please enter a longer news article (at least 10 characters)"})
        
        result = predict_news(news_text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"})

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
