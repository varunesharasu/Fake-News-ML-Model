import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer

def fix_and_retrain_model():
    """Fix label issues and retrain the model"""
    
    print("🔧 Fixing and Retraining Model...")
    
    # Initialize components
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    
    # Load data
    print("📊 Loading data...")
    df = data_processor.load_data('data/raw/Fake.csv', 'data/raw/True.csv')
    
    if df is None:
        print("❌ Failed to load data!")
        return
    
    # Check label distribution
    print(f"\n📈 Original Label Distribution:")
    print(f"Label 0 (Real): {len(df[df['label'] == 0])}")
    print(f"Label 1 (Fake): {len(df[df['label'] == 1])}")
    
    # Sample balanced data
    print("\n⚖️ Creating balanced sample...")
    fake_sample = df[df['label'] == 1].sample(n=3000, random_state=42)
    real_sample = df[df['label'] == 0].sample(n=3000, random_state=42)
    df_balanced = pd.concat([fake_sample, real_sample]).reset_index(drop=True)
    
    print(f"Balanced dataset shape: {df_balanced.shape}")
    print("Balanced label distribution:")
    print(df_balanced['label'].value_counts().sort_index())
    
    # Quick preprocessing
    print("\n🧹 Preprocessing...")
    processed_df = df_balanced.copy()
    
    # Combine title and text
    if 'title' in df_balanced.columns and 'text' in df_balanced.columns:
        processed_df['content'] = df_balanced['title'].fillna('') + ' ' + df_balanced['text'].fillna('')
    elif 'title' in df_balanced.columns:
        processed_df['content'] = df_balanced['title'].fillna('')
    elif 'text' in df_balanced.columns:
        processed_df['content'] = df_balanced['text'].fillna('')
    
    # Clean and process text
    processed_df['cleaned_content'] = processed_df['content'].apply(data_processor.clean_text)
    processed_df['content_no_stopwords'] = processed_df['cleaned_content'].apply(data_processor.remove_stopwords)
    processed_df['lemmatized_content'] = processed_df['content_no_stopwords']
    
    # Extract sentiment features
    sentiment_features = processed_df['cleaned_content'].apply(data_processor.get_sentiment_features)
    sentiment_df = pd.json_normalize(sentiment_features)
    processed_df = pd.concat([processed_df, sentiment_df], axis=1)
    
    # Feature engineering
    print("\n⚙️ Feature engineering...")
    features_df = feature_engineer.extract_text_features(processed_df)
    
    # Prepare features
    feature_columns = ['char_count', 'word_count', 'sentence_count', 'avg_word_length',
                      'uppercase_ratio', 'punctuation_count', 'exclamation_count', 
                      'question_count', 'textblob_polarity', 'textblob_subjectivity',
                      'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral']
    
    X_features = features_df[feature_columns].fillna(0)
    y = features_df['label']
    
    print(f"\n📊 Feature Statistics:")
    print(f"Features shape: {X_features.shape}")
    print(f"Target distribution: {y.value_counts().sort_index().to_dict()}")
    
    # Split data
    X_train_feat, X_test_feat, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✂️ Data Split:")
    print(f"Train set: {X_train_feat.shape[0]} samples")
    print(f"Test set: {X_test_feat.shape[0]} samples")
    print(f"Train labels: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Test labels: {y_test.value_counts().sort_index().to_dict()}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_feat_scaled = scaler.fit_transform(X_train_feat)
    X_test_feat_scaled = scaler.transform(X_test_feat)
    
    # Prepare text data
    train_texts = features_df.loc[X_train_feat.index, 'content_no_stopwords'].fillna('')
    test_texts = features_df.loc[X_test_feat.index, 'content_no_stopwords'].fillna('')
    
    # Create TF-IDF features
    print("\n📝 Creating TF-IDF features...")
    tfidf_train = feature_engineer.create_tfidf_features(train_texts, max_features=2000)
    tfidf_test = feature_engineer.tfidf_vectorizer.transform(test_texts)
    
    # Combine features
    X_train_combined = np.hstack([X_train_feat_scaled, tfidf_train.toarray()])
    X_test_combined = np.hstack([X_test_feat_scaled, tfidf_test.toarray()])
    
    print(f"Combined features shape: {X_train_combined.shape}")
    
    # Train LightGBM model
    print("\n🤖 Training LightGBM model...")
    from lightgbm import LGBMClassifier
    
    model = LGBMClassifier(
        random_state=42, 
        verbose=-1,
        class_weight='balanced'  # Handle any remaining imbalance
    )
    
    model.fit(X_train_combined, y_train)
    
    # Test predictions
    print("\n🧪 Testing model...")
    y_pred = model.predict(X_test_combined)
    y_pred_proba = model.predict_proba(X_test_combined)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Detailed evaluation
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔍 Confusion Matrix:")
    print(f"True Real, Pred Real: {cm[0,0]}")
    print(f"True Real, Pred Fake: {cm[0,1]}")
    print(f"True Fake, Pred Real: {cm[1,0]}")
    print(f"True Fake, Pred Fake: {cm[1,1]}")
    
    # Test with sample predictions
    print(f"\n🎯 Sample Predictions:")
    for i in range(min(10, len(y_test))):
        actual = "Real" if y_test.iloc[i] == 0 else "Fake"
        predicted = "Real" if y_pred[i] == 0 else "Fake"
        confidence = max(y_pred_proba[i]) * 100
        print(f"  Actual: {actual}, Predicted: {predicted}, Confidence: {confidence:.1f}%")
    
    # Save models
    print("\n💾 Saving corrected models...")
    os.makedirs('models/saved_models', exist_ok=True)
    
    joblib.dump(model, 'models/saved_models/lightgbm_model.pkl')
    joblib.dump(model, 'models/saved_models/best_model.pkl')
    joblib.dump(scaler, 'models/saved_models/scaler.pkl')
    joblib.dump(feature_engineer, 'models/saved_models/feature_engineer.pkl')
    
    # Save model info
    model_info = {
        'model_type': 'LightGBM',
        'accuracy': float(accuracy),
        'training_samples': len(df_balanced),
        'features': feature_columns,
        'label_mapping': {'0': 'Real', '1': 'Fake'},
        'balanced_training': True
    }
    
    import json
    with open('models/saved_models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Models saved successfully!")
    print("\n🚀 Now test with: python debug_model.py")

if __name__ == "__main__":
    fix_and_retrain_model()
