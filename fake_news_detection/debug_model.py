import joblib
import pandas as pd
import numpy as np
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
import os

def debug_model():
    """Debug the model predictions"""
    
    print("🔍 Debugging Model Predictions...")
    
    # Load models
    try:
        model = joblib.load('models/saved_models/lightgbm_model.pkl')
        scaler = joblib.load('models/saved_models/scaler.pkl')
        feature_engineer = joblib.load('models/saved_models/feature_engineer.pkl')
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Initialize processors
    data_processor = DataProcessor()
    
    # Test with known real and fake news
    test_cases = [
        {
            'text': "The Federal Reserve announced today that it will maintain current interest rates at 5.25% following their monthly meeting. The decision comes amid concerns about inflation and economic stability in global markets.",
            'expected': 'REAL'
        },
        {
            'text': "Scientists at MIT have developed a new method for detecting cancer cells using artificial intelligence. The research, published in Nature Medicine, shows promising results in early detection of various cancer types.",
            'expected': 'REAL'
        },
        {
            'text': "BREAKING: Government officials confirm that aliens have been living among us for decades! Secret documents reveal shocking truth about extraterrestrial beings controlling world governments. Click here to learn more!",
            'expected': 'FAKE'
        },
        {
            'text': "You won't believe what happens next! This one weird trick will make you rich overnight! Doctors hate this simple method that cures everything!",
            'expected': 'FAKE'
        }
    ]
    
    print("\n📊 Testing Model Predictions:")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i} (Expected: {test_case['expected']}):")
        print(f"Text: {test_case['text'][:100]}...")
        
        try:
            result = predict_news_debug(test_case['text'], model, scaler, feature_engineer, data_processor)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                prediction = result['prediction']
                real_conf = result['real_confidence']
                fake_conf = result['fake_confidence']
                
                print(f"🔮 Prediction: {prediction}")
                print(f"📈 Real Confidence: {real_conf}%")
                print(f"📈 Fake Confidence: {fake_conf}%")
                
                # Check if prediction matches expected
                if prediction == test_case['expected']:
                    print("✅ CORRECT")
                else:
                    print("❌ INCORRECT")
                    
                # Show raw model output
                print(f"🔧 Raw model prediction: {result.get('raw_prediction', 'N/A')}")
                print(f"🔧 Raw probabilities: {result.get('raw_probabilities', 'N/A')}")
                
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
        
        print("-" * 80)
    
    # Check model info
    try:
        import json
        with open('models/saved_models/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        print(f"\n📋 Model Information:")
        print(f"Best Model: {model_info.get('best_model', 'Unknown')}")
        print(f"Training Samples: {model_info.get('training_samples', 'Unknown')}")
        print(f"Total Features: {model_info.get('total_features', 'Unknown')}")
        
        if 'model_performance' in model_info:
            print(f"\n📊 Model Performance:")
            for perf in model_info['model_performance'][:3]:  # Top 3
                print(f"  {perf['Model']}: Accuracy = {perf['Accuracy']:.4f}")
                
    except Exception as e:
        print(f"⚠️ Could not load model info: {e}")

def predict_news_debug(news_text, model, scaler, feature_engineer, data_processor):
    """Debug version of predict_news with detailed logging"""
    
    try:
        # Create a temporary dataframe
        temp_df = pd.DataFrame({
            'content': [news_text],
            'label': [0]  # Dummy label
        })
        
        print(f"  📝 Original text length: {len(news_text)} characters")
        
        # Preprocess the text
        temp_df['cleaned_content'] = temp_df['content'].apply(data_processor.clean_text)
        temp_df['content_no_stopwords'] = temp_df['cleaned_content'].apply(data_processor.remove_stopwords)
        temp_df['lemmatized_content'] = temp_df['content_no_stopwords']
        
        print(f"  🧹 Cleaned text length: {len(temp_df['cleaned_content'].iloc[0])} characters")
        
        # Extract sentiment features
        sentiment_features = temp_df['cleaned_content'].apply(data_processor.get_sentiment_features)
        sentiment_df = pd.json_normalize(sentiment_features)
        temp_df = pd.concat([temp_df, sentiment_df], axis=1)
        
        print(f"  😊 Sentiment polarity: {sentiment_df['textblob_polarity'].iloc[0]:.3f}")
        print(f"  😊 Sentiment subjectivity: {sentiment_df['textblob_subjectivity'].iloc[0]:.3f}")
        
        # Extract text features
        features_df = feature_engineer.extract_text_features(temp_df)
        
        # Select the same features used in training
        feature_columns = ['char_count', 'word_count', 'sentence_count', 'avg_word_length',
                          'uppercase_ratio', 'punctuation_count', 'exclamation_count', 
                          'question_count', 'textblob_polarity', 'textblob_subjectivity',
                          'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral']
        
        X_features = features_df[feature_columns].fillna(0)
        
        print(f"  📊 Word count: {X_features['word_count'].iloc[0]}")
        print(f"  📊 Exclamation count: {X_features['exclamation_count'].iloc[0]}")
        print(f"  📊 Question count: {X_features['question_count'].iloc[0]}")
        
        # Create TF-IDF features
        if hasattr(feature_engineer, 'tfidf_vectorizer') and feature_engineer.tfidf_vectorizer is not None:
            tfidf_features = feature_engineer.tfidf_vectorizer.transform([temp_df['content_no_stopwords'].iloc[0]])
            print(f"  🔤 TF-IDF features shape: {tfidf_features.shape}")
        else:
            return {"error": "TF-IDF vectorizer not available"}
        
        # Scale the basic features
        X_features_scaled = scaler.transform(X_features)
        
        # Combine features
        X_combined = np.hstack([X_features_scaled, tfidf_features.toarray()])
        print(f"  🔧 Combined features shape: {X_combined.shape}")
        
        # Make prediction
        raw_prediction = model.predict(X_combined)[0]
        raw_probabilities = model.predict_proba(X_combined)[0]
        
        print(f"  🎯 Raw prediction: {raw_prediction}")
        print(f"  🎯 Raw probabilities: [Real: {raw_probabilities[0]:.4f}, Fake: {raw_probabilities[1]:.4f}]")
        
        # Get confidence scores
        fake_confidence = raw_probabilities[1] * 100
        real_confidence = raw_probabilities[0] * 100
        
        result = {
            'prediction': 'FAKE' if raw_prediction == 1 else 'REAL',
            'fake_confidence': round(fake_confidence, 2),
            'real_confidence': round(real_confidence, 2),
            'raw_prediction': int(raw_prediction),
            'raw_probabilities': raw_probabilities.tolist(),
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

if __name__ == "__main__":
    debug_model()
