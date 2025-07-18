import pandas as pd
import numpy as np
import joblib
import os
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer

def diagnose_everything():
    """Complete diagnosis of the entire pipeline"""
    
    print("🔍 COMPLETE SYSTEM DIAGNOSIS")
    print("="*60)
    
    # Step 1: Check raw data
    print("\n1. 📊 CHECKING RAW DATA...")
    try:
        # Load raw data directly
        fake_df = pd.read_csv('data/raw/Fake.csv')
        true_df = pd.read_csv('data/raw/True.csv')
        
        print(f"✅ Fake news CSV: {len(fake_df)} articles")
        print(f"✅ True news CSV: {len(true_df)} articles")
        print(f"Fake CSV columns: {list(fake_df.columns)}")
        print(f"True CSV columns: {list(true_df.columns)}")
        
        # Check first few samples
        print(f"\nFirst fake news sample:")
        if 'title' in fake_df.columns:
            print(f"Title: {fake_df['title'].iloc[0][:100]}...")
        if 'text' in fake_df.columns:
            print(f"Text: {fake_df['text'].iloc[0][:100]}...")
            
        print(f"\nFirst true news sample:")
        if 'title' in true_df.columns:
            print(f"Title: {true_df['title'].iloc[0][:100]}...")
        if 'text' in true_df.columns:
            print(f"Text: {true_df['text'].iloc[0][:100]}...")
            
    except Exception as e:
        print(f"❌ Error loading raw data: {e}")
        return
    
    # Step 2: Check data processor
    print("\n2. 🧹 CHECKING DATA PROCESSOR...")
    try:
        data_processor = DataProcessor()
        combined_df = data_processor.load_data('data/raw/Fake.csv', 'data/raw/True.csv')
        
        if combined_df is not None:
            print(f"✅ Combined dataset: {len(combined_df)} articles")
            print(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")
            
            # Check what labels mean
            fake_sample = combined_df[combined_df['label'] == 1].iloc[0]
            real_sample = combined_df[combined_df['label'] == 0].iloc[0]
            
            print(f"\nLabel 1 sample (should be FAKE):")
            print(f"Content: {fake_sample.get('content', fake_sample.get('title', 'N/A'))[:100]}...")
            
            print(f"\nLabel 0 sample (should be REAL):")
            print(f"Content: {real_sample.get('content', real_sample.get('title', 'N/A'))[:100]}...")
            
        else:
            print("❌ Failed to load combined data")
            return
            
    except Exception as e:
        print(f"❌ Error in data processor: {e}")
        return
    
    # Step 3: Test preprocessing
    print("\n3. 🔧 TESTING PREPROCESSING...")
    try:
        test_texts = [
            "The Federal Reserve announced interest rate changes today.",
            "BREAKING: Aliens control the government! Click here!"
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\nTest text {i+1}: {text}")
            
            # Test cleaning
            cleaned = data_processor.clean_text(text)
            print(f"Cleaned: {cleaned}")
            
            # Test stopword removal
            no_stopwords = data_processor.remove_stopwords(cleaned)
            print(f"No stopwords: {no_stopwords}")
            
            # Test sentiment
            sentiment = data_processor.get_sentiment_features(cleaned)
            print(f"Sentiment: {sentiment}")
            
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
    
    # Step 4: Check if models exist and test them
    print("\n4. 🤖 CHECKING MODELS...")
    model_files = [
        'models/saved_models/lightgbm_model.pkl',
        'models/saved_models/best_model.pkl',
        'models/saved_models/scaler.pkl',
        'models/saved_models/feature_engineer.pkl'
    ]
    
    for file in model_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
    
    # Step 5: Test model if available
    if all(os.path.exists(f) for f in model_files):
        print("\n5. 🧪 TESTING MODEL...")
        try:
            model = joblib.load('models/saved_models/lightgbm_model.pkl')
            scaler = joblib.load('models/saved_models/scaler.pkl')
            feature_engineer = joblib.load('models/saved_models/feature_engineer.pkl')
            
            # Test with simple examples
            test_cases = [
                "The stock market closed higher today after positive economic news.",
                "SHOCKING: This one weird trick will make you rich overnight!"
            ]
            
            for text in test_cases:
                result = test_single_prediction(text, model, scaler, feature_engineer, data_processor)
                print(f"\nText: {text[:50]}...")
                print(f"Result: {result}")
                
        except Exception as e:
            print(f"❌ Error testing model: {e}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")

def test_single_prediction(text, model, scaler, feature_engineer, data_processor):
    """Test a single prediction with full debugging"""
    try:
        # Create dataframe
        temp_df = pd.DataFrame({'content': [text], 'label': [0]})
        
        # Preprocess
        temp_df['cleaned_content'] = temp_df['content'].apply(data_processor.clean_text)
        temp_df['content_no_stopwords'] = temp_df['cleaned_content'].apply(data_processor.remove_stopwords)
        temp_df['lemmatized_content'] = temp_df['content_no_stopwords']
        
        # Sentiment
        sentiment_features = temp_df['cleaned_content'].apply(data_processor.get_sentiment_features)
        sentiment_df = pd.json_normalize(sentiment_features)
        temp_df = pd.concat([temp_df, sentiment_df], axis=1)
        
        # Features
        features_df = feature_engineer.extract_text_features(temp_df)
        
        feature_columns = ['char_count', 'word_count', 'sentence_count', 'avg_word_length',
                          'uppercase_ratio', 'punctuation_count', 'exclamation_count', 
                          'question_count', 'textblob_polarity', 'textblob_subjectivity',
                          'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral']
        
        X_features = features_df[feature_columns].fillna(0)
        
        # TF-IDF
        tfidf_features = feature_engineer.tfidf_vectorizer.transform([temp_df['content_no_stopwords'].iloc[0]])
        
        # Scale and combine
        X_features_scaled = scaler.transform(X_features)
        X_combined = np.hstack([X_features_scaled, tfidf_features.toarray()])
        
        # Predict
        raw_pred = model.predict(X_combined)[0]
        raw_proba = model.predict_proba(X_combined)[0]
        
        return {
            'raw_prediction': int(raw_pred),
            'probabilities': raw_proba.tolist(),
            'interpretation': 'FAKE' if raw_pred == 1 else 'REAL',
            'confidence': f"Real: {raw_proba[0]*100:.1f}%, Fake: {raw_proba[1]*100:.1f}%"
        }
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    diagnose_everything()
