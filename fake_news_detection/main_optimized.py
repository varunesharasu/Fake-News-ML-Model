import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.models import TraditionalModels, DeepLearningModels
from src.evaluation import ModelEvaluator

def main():
    print("=== Optimized Fake News Detection Pipeline ===")
    
    # Initialize components
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    evaluator = ModelEvaluator()
    
    # Step 1: Load and sample data for faster processing
    print("\n1. Loading and preprocessing data...")
    df = data_processor.load_data('data/raw/Fake.csv', 'data/raw/True.csv')
    
    if df is None:
        print("Failed to load data!")
        return
    
    # Sample data for faster processing (optional)
    print("Sampling data for faster processing...")
    df_sampled = df.groupby('label').apply(lambda x: x.sample(min(5000, len(x)), random_state=42)).reset_index(drop=True)
    print(f"Sampled dataset shape: {df_sampled.shape}")
    print(f"Label distribution:\n{df_sampled['label'].value_counts()}")
    
    # Quick preprocessing without lemmatization
    print("Quick preprocessing...")
    processed_df = quick_preprocess(df_sampled, data_processor)
    
    # Step 2: Feature Engineering
    print("\n2. Feature engineering...")
    features_df = feature_engineer.extract_text_features(processed_df)
    
    # Step 3: Prepare data for modeling
    print("\n3. Preparing data for modeling...")
    
    # Select features for traditional ML
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
    X_test_feat_scaled = scaler.transform(X_test_feat)
    
    # Prepare text data (using cleaned content instead of lemmatized)
    train_texts = features_df.loc[X_train_feat.index, 'content_no_stopwords'].fillna('')
    test_texts = features_df.loc[X_test_feat.index, 'content_no_stopwords'].fillna('')
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf_train, tfidf_test = feature_engineer.create_tfidf_features(
        train_texts, test_texts, max_features=3000  # Reduced for speed
    )
    
    # Combine features
    X_train_combined = np.hstack([X_train_feat_scaled, tfidf_train.toarray()])
    X_test_combined = np.hstack([X_test_feat_scaled, tfidf_test.toarray()])
    
    print(f"Combined feature shape: {X_train_combined.shape}")
    
    # Step 4: Train Traditional ML Models
    print("\n4. Training traditional ML models...")
    
    traditional_ml = TraditionalModels()
    traditional_ml.initialize_models()
    trained_models = traditional_ml.train_individual_models(X_train_combined, y_train)
    
    # Evaluate traditional models
    print("\n5. Evaluating traditional models...")
    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test_combined)
            y_pred_proba = model.predict_proba(X_test_combined)[:, 1]
            
            results = evaluator.evaluate_model(y_test, y_pred, y_pred_proba, name)
            print(f"{name}: Accuracy = {results['Accuracy']:.4f}, F1-Score = {results['F1-Score']:.4f}, AUC = {results['AUC']:.4f}")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    # Train ensemble model
    print("\n6. Training ensemble model...")
    try:
        ensemble_model = traditional_ml.create_ensemble(X_train_combined, y_train)
        y_pred_ensemble = ensemble_model.predict(X_test_combined)
        y_pred_proba_ensemble = ensemble_model.predict_proba(X_test_combined)[:, 1]
        
        ensemble_results = evaluator.evaluate_model(
            y_test, y_pred_ensemble, y_pred_proba_ensemble, "Ensemble"
        )
        print(f"Ensemble: Accuracy = {ensemble_results['Accuracy']:.4f}, F1-Score = {ensemble_results['F1-Score']:.4f}, AUC = {ensemble_results['AUC']:.4f}")
    except Exception as e:
        print(f"Error with ensemble model: {e}")
    
    # Step 7: Quick Deep Learning Model
    print("\n7. Training simple deep learning model...")
    try:
        dl_models = DeepLearningModels()
        
        # Prepare sequences for deep learning (smaller vocab for speed)
        all_texts = list(train_texts) + list(test_texts)
        sequences = dl_models.prepare_sequences(all_texts, vocab_size=5000, max_length=100)
        
        train_sequences = sequences[:len(train_texts)]
        test_sequences = sequences[len(train_texts):]
        
        # Train CNN model with fewer epochs
        print("Training CNN model...")
        cnn_model = dl_models.create_cnn_model(vocab_size=5000, max_length=100)
        
        history_cnn = cnn_model.fit(
            train_sequences, y_train,
            validation_data=(test_sequences, y_test),
            epochs=3,  # Reduced epochs for speed
            batch_size=64,  # Larger batch size for speed
            verbose=1
        )
        
        # Evaluate CNN
        y_pred_cnn_proba = cnn_model.predict(test_sequences).flatten()
        y_pred_cnn = (y_pred_cnn_proba > 0.5).astype(int)
        
        cnn_results = evaluator.evaluate_model(
            y_test, y_pred_cnn, y_pred_cnn_proba, "CNN"
        )
        print(f"CNN: Accuracy = {cnn_results['Accuracy']:.4f}, F1-Score = {cnn_results['F1-Score']:.4f}, AUC = {cnn_results['AUC']:.4f}")
        
    except Exception as e:
        print(f"Error with deep learning model: {e}")
    
    # Step 8: Model Comparison and Visualization
    print("\n8. Model comparison and visualization...")
    
    # Display results table
    results_df = evaluator.compare_models()
    if results_df is not None:
        print("\nModel Comparison Results:")
        print(results_df.to_string(index=False))
        
        # Plot comparisons
        evaluator.plot_model_comparison()
        
        # Plot confusion matrix for best model
        best_model_name = results_df.iloc[0]['Model']
        print(f"\nBest performing model: {best_model_name}")
        
        # Save best model
        import joblib
        import os
        os.makedirs('models/saved_models', exist_ok=True)
        
        if best_model_name in trained_models:
            joblib.dump(trained_models[best_model_name], f'models/saved_models/{best_model_name}_model.pkl')
            joblib.dump(scaler, 'models/saved_models/scaler.pkl')
            joblib.dump(feature_engineer, 'models/saved_models/feature_engineer.pkl')
            print(f"\nBest model ({best_model_name}) saved successfully!")
    
    print("\n=== Pipeline Complete ===")

def quick_preprocess(df, data_processor):
    """Quick preprocessing without lemmatization"""
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
    processed_df['cleaned_content'] = processed_df['content'].apply(data_processor.clean_text)
    
    # Remove stopwords
    print("Removing stopwords...")
    processed_df['content_no_stopwords'] = processed_df['cleaned_content'].apply(data_processor.remove_stopwords)
    
    # Skip lemmatization for speed, use content_no_stopwords instead
    processed_df['lemmatized_content'] = processed_df['content_no_stopwords']
    
    # Extract sentiment features
    print("Extracting sentiment features...")
    sentiment_features = processed_df['cleaned_content'].apply(data_processor.get_sentiment_features)
    sentiment_df = pd.json_normalize(sentiment_features)
    
    # Combine with main dataframe
    processed_df = pd.concat([processed_df, sentiment_df], axis=1)
    
    return processed_df

if __name__ == "__main__":
    main()