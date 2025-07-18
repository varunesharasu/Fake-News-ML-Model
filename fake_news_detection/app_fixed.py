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

# Global variables for models
model = None
scaler = None
feature_engineer = None
data_processor = DataProcessor()

def load_models():
    """Load the trained models"""
    global model, scaler, feature_engineer
    
    try:
        # Load model
        model_files = [
            'models/saved_models/lightgbm_model.pkl',
            'models/saved_models/best_model.pkl'
        ]
        
        model_loaded = False
        for model_file in model_files:
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                print(f"✅ Model loaded from: {model_file}")
                model_loaded = True
                break
        
        if not model_loaded:
            print("❌ No model file found!")
            return False
        
        # Load preprocessors
        scaler = joblib.load('models/saved_models/scaler.pkl')
        feature_engineer = joblib.load('models/saved_models/feature_engineer.pkl')
        print("✅ Preprocessors loaded!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def predict_news(news_text):
    """Predict if news is fake or real with improved logic"""
    if model is None or scaler is None or feature_engineer is None:
        return {"error": "Models not loaded. Please train the models first."}
    
    try:
        # Input validation
        if len(news_text.strip()) < 10:
            return {"error": "Please enter a longer news article (at least 10 characters)."}
        
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
        if hasattr(feature_engineer, 'tfidf_vectorizer') and feature_engineer.tfidf_vectorizer is not None:
            tfidf_features = feature_engineer.tfidf_vectorizer.transform([temp_df['content_no_stopwords'].iloc[0]])
        else:
            return {"error": "TF-IDF vectorizer not available"}
        
        # Scale the basic features
        X_features_scaled = scaler.transform(X_features)
        
        # Combine features
        X_combined = np.hstack([X_features_scaled, tfidf_features.toarray()])
        
        # Make prediction
        raw_prediction = model.predict(X_combined)[0]
        raw_probabilities = model.predict_proba(X_combined)[0]
        
        # Ensure we have the right mapping: 0 = Real, 1 = Fake
        real_confidence = raw_probabilities[0] * 100
        fake_confidence = raw_probabilities[1] * 100
        
        # Determine prediction based on raw model output
        prediction_label = 'FAKE' if raw_prediction == 1 else 'REAL'
        
        # Additional validation - check if probabilities make sense
        if real_confidence + fake_confidence < 95:  # Should sum to ~100%
            return {"error": "Model prediction uncertainty detected"}
        
        result = {
            'prediction': prediction_label,
            'fake_confidence': round(fake_confidence, 2),
            'real_confidence': round(real_confidence, 2),
            'features': {
                'word_count': int(features_df['word_count'].iloc[0]),
                'char_count': int(features_df['char_count'].iloc[0]),
                'sentiment_polarity': round(features_df['textblob_polarity'].iloc[0], 3),
                'sentiment_subjectivity': round(features_df['textblob_subjectivity'].iloc[0], 3)
            },
            'debug_info': {
                'raw_prediction': int(raw_prediction),
                'raw_probabilities': [round(p, 4) for p in raw_probabilities]
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
        
        result = predict_news(news_text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"})

@app.route('/about')
def about():
    return render_template('about.html')

# Load models when the app starts
models_loaded = load_models()

if __name__ == '__main__':
    if not models_loaded:
        print("\n" + "="*50)
        print("⚠️  MODELS NOT LOADED!")
        print("Please run: python fix_model_training.py")
        print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
