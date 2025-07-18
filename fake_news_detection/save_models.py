import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.models import TraditionalModels

def quick_train_and_save():
    """Quick training and saving of the best model"""
    
    print("🚀 Quick Model Training and Saving...")
    
    # Initialize components
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    
    # Load and preprocess data
    print("📊 Loading data...")
    df = data_processor.load_data('data/raw/Fake.csv', 'data/raw/True.csv')
    
    if df is None:
        print("❌ Failed to load data!")
        return
    
    # Sample data for faster processing
    print("🔄 Sampling data...")
    df_sampled = df.groupby('label').apply(lambda x: x.sample(min(3000, len(x)), random_state=42)).reset_index(drop=True)
    
    # Quick preprocessing
    print("🧹 Preprocessing...")
    processed_df = quick_preprocess(df_sampled, data_processor)
    
    # Feature engineering
    print("⚙️ Feature engineering...")
    features_df = feature_engineer.extract_text_features(processed_df)
    
    # Prepare features
    feature_columns = ['char_count', 'word_count', 'sentence_count', 'avg_word_length',
                      'uppercase_ratio', 'punctuation_count', 'exclamation_count', 
                      'question_count', 'textblob_polarity', 'textblob_subjectivity',
                      'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral']
    
    X_features = features_df[feature_columns].fillna(0)
    y = features_df['label']
    
    # Split data
    X_train_feat, X_test_feat, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_feat_scaled = scaler.fit_transform(X_train_feat)
    
    # Prepare text data
    train_texts = features_df.loc[X_train_feat.index, 'content_no_stopwords'].fillna('')
    
    # Create TF-IDF features
    print("📝 Creating TF-IDF features...")
    tfidf_train = feature_engineer.create_tfidf_features(train_texts, max_features=2000)
    
    # Combine features
    X_train_combined = np.hstack([X_train_feat_scaled, tfidf_train.toarray()])
    
    # Train LightGBM (best performing model)
    print("🤖 Training LightGBM model...")
    from lightgbm import LGBMClassifier
    
    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train_combined, y_train)
    
    # Create directories
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Save models
    print("💾 Saving models...")
    joblib.dump(model, 'models/saved_models/lightgbm_model.pkl')
    joblib.dump(model, 'models/saved_models/best_model.pkl')
    joblib.dump(scaler, 'models/saved_models/scaler.pkl')
    joblib.dump(feature_engineer, 'models/saved_models/feature_engineer.pkl')
    
    print("✅ Models saved successfully!")
    print("📁 Files saved:")
    print("   - models/saved_models/lightgbm_model.pkl")
    print("   - models/saved_models/best_model.pkl")
    print("   - models/saved_models/scaler.pkl")
    print("   - models/saved_models/feature_engineer.pkl")
    
    return True

def quick_preprocess(df, data_processor):
    """Quick preprocessing without lemmatization"""
    processed_df = df.copy()
    
    # Combine title and text
    if 'title' in df.columns and 'text' in df.columns:
        processed_df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'title' in df.columns:
        processed_df['content'] = df['title'].fillna('')
    elif 'text' in df.columns:
        processed_df['content'] = df['text'].fillna('')
    
    # Clean text
    processed_df['cleaned_content'] = processed_df['content'].apply(data_processor.clean_text)
    processed_df['content_no_stopwords'] = processed_df['cleaned_content'].apply(data_processor.remove_stopwords)
    processed_df['lemmatized_content'] = processed_df['content_no_stopwords']
    
    # Extract sentiment features
    sentiment_features = processed_df['cleaned_content'].apply(data_processor.get_sentiment_features)
    sentiment_df = pd.json_normalize(sentiment_features)
    processed_df = pd.concat([processed_df, sentiment_df], axis=1)
    
    return processed_df

if __name__ == "__main__":
    quick_train_and_save()