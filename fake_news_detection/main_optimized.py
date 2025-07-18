# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')

# # Import custom modules
# from src.data_processing import DataProcessor
# from src.feature_engineering import FeatureEngineer
# from src.models import TraditionalModels, DeepLearningModels
# from src.evaluation import ModelEvaluator

# def main():
#     print("=== Optimized Fake News Detection Pipeline ===")
    
#     # Initialize components
#     data_processor = DataProcessor()
#     feature_engineer = FeatureEngineer()
#     evaluator = ModelEvaluator()
    
#     # Step 1: Load and sample data for faster processing
#     print("\n1. Loading and preprocessing data...")
#     df = data_processor.load_data('data/raw/Fake.csv', 'data/raw/True.csv')
    
#     if df is None:
#         print("Failed to load data!")
#         return
    
#     # Sample data for faster processing (optional)
#     print("Sampling data for faster processing...")
#     df_sampled = df.groupby('label').apply(lambda x: x.sample(min(5000, len(x)), random_state=42)).reset_index(drop=True)
#     print(f"Sampled dataset shape: {df_sampled.shape}")
#     print(f"Label distribution:\n{df_sampled['label'].value_counts()}")
    
#     # Quick preprocessing without lemmatization
#     print("Quick preprocessing...")
#     processed_df = quick_preprocess(df_sampled, data_processor)
    
#     # Step 2: Feature Engineering
#     print("\n2. Feature engineering...")
#     features_df = feature_engineer.extract_text_features(processed_df)
    
#     # Step 3: Prepare data for modeling
#     print("\n3. Preparing data for modeling...")
    
#     # Select features for traditional ML
#     feature_columns = ['char_count', 'word_count', 'sentence_count', 'avg_word_length',
#                       'uppercase_ratio', 'punctuation_count', 'exclamation_count', 
#                       'question_count', 'textblob_polarity', 'textblob_subjectivity',
#                       'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral']
    
#     X_features = features_df[feature_columns].fillna(0)
#     y = features_df['label']
    
#     # Split data
#     X_train_feat, X_test_feat, y_train, y_test = train_test_split(
#         X_features, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_feat_scaled = scaler.fit_transform(X_train_feat)
#     X_test_feat_scaled = scaler.transform(X_test_feat)
    
#     # Prepare text data (using cleaned content instead of lemmatized)
#     train_texts = features_df.loc[X_train_feat.index, 'content_no_stopwords'].fillna('')
#     test_texts = features_df.loc[X_test_feat.index, 'content_no_stopwords'].fillna('')
    
#     # Create TF-IDF features
#     print("Creating TF-IDF features...")
#     tfidf_train, tfidf_test = feature_engineer.create_tfidf_features(
#         train_texts, test_texts, max_features=3000  # Reduced for speed
#     )
    
#     # Combine features
#     X_train_combined = np.hstack([X_train_feat_scaled, tfidf_train.toarray()])
#     X_test_combined = np.hstack([X_test_feat_scaled, tfidf_test.toarray()])
    
#     print(f"Combined feature shape: {X_train_combined.shape}")
    
#     # Step 4: Train Traditional ML Models
#     print("\n4. Training traditional ML models...")
    
#     traditional_ml = TraditionalModels()
#     traditional_ml.initialize_models()
#     trained_models = traditional_ml.train_individual_models(X_train_combined, y_train)
    
#     # Evaluate traditional models
#     print("\n5. Evaluating traditional models...")
#     predictions = traditional_ml.predict_individual_models(trained_models, X_test_combined)
    
#     for name, pred_dict in predictions.items():
#         try:
#             results = evaluator.evaluate_model(
#                 y_test, pred_dict['pred'], pred_dict['pred_proba'], name
#             )
#             print(f"{name}: Accuracy = {results['Accuracy']:.4f}, F1-Score = {results['F1-Score']:.4f}, AUC = {results['AUC']:.4f}")
#         except Exception as e:
#             print(f"Error evaluating {name}: {e}")
    
#     # Train ensemble model
#     print("\n6. Training ensemble model...")
#     try:
#         ensemble_model = traditional_ml.create_ensemble(X_train_combined, y_train)
#         y_pred_ensemble = ensemble_model.predict(X_test_combined)
#         y_pred_proba_ensemble = ensemble_model.predict_proba(X_test_combined)[:, 1]
        
#         ensemble_results = evaluator.evaluate_model(
#             y_test, y_pred_ensemble, y_pred_proba_ensemble, "Ensemble"
#         )
#         print(f"Ensemble: Accuracy = {ensemble_results['Accuracy']:.4f}, F1-Score = {ensemble_results['F1-Score']:.4f}, AUC = {ensemble_results['AUC']:.4f}")
#     except Exception as e:
#         print(f"Error with ensemble model: {e}")
    
#     # Step 7: Quick Deep Learning Model
#     print("\n7. Training simple deep learning model...")
#     try:
#         dl_models = DeepLearningModels()
        
#         if hasattr(dl_models, 'tokenizer'):  # Check if TensorFlow is available
#             # Prepare sequences for deep learning (smaller vocab for speed)
#             all_texts = list(train_texts) + list(test_texts)
#             sequences = dl_models.prepare_sequences(all_texts, vocab_size=5000, max_length=100)
            
#             if sequences is not None:
#                 train_sequences = sequences[:len(train_texts)]
#                 test_sequences = sequences[len(train_texts):]
                
#                 # Train CNN model with fewer epochs
#                 print("Training CNN model...")
#                 cnn_model = dl_models.create_cnn_model(vocab_size=5000, max_length=100)
                
#                 if cnn_model is not None:
#                     history_cnn = cnn_model.fit(
#                         train_sequences, y_train,
#                         validation_data=(test_sequences, y_test),
#                         epochs=3,  # Reduced epochs for speed
#                         batch_size=64,  # Larger batch size for speed
#                         verbose=1
#                     )
                    
#                     # Evaluate CNN
#                     y_pred_cnn_proba = cnn_model.predict(test_sequences).flatten()
#                     y_pred_cnn = (y_pred_cnn_proba > 0.5).astype(int)
                    
#                     cnn_results = evaluator.evaluate_model(
#                         y_test, y_pred_cnn, y_pred_cnn_proba, "CNN"
#                     )
#                     print(f"CNN: Accuracy = {cnn_results['Accuracy']:.4f}, F1-Score = {cnn_results['F1-Score']:.4f}, AUC = {cnn_results['AUC']:.4f}")
#         else:
#             print("Skipping deep learning models - TensorFlow not available")
        
#     except Exception as e:
#         print(f"Error with deep learning model: {e}")
    
#     # Step 8: Model Comparison and Visualization
#     print("\n8. Model comparison and visualization...")
    
#     # Display results table
#     results_df = evaluator.compare_models()
#     if results_df is not None and len(results_df) > 0:
#         print("\nModel Comparison Results:")
#         print(results_df.to_string(index=False))
        
#         # Plot comparisons
#         try:
#             evaluator.plot_model_comparison()
#         except Exception as e:
#             print(f"Error plotting comparison: {e}")
        
#         # Plot confusion matrix for best model
#         best_model_name = results_df.iloc[0]['Model']
#         print(f"\nBest performing model: {best_model_name}")
        
#         # Save best model
#         import joblib
#         import os
#         os.makedirs('models/saved_models', exist_ok=True)
        
#         if best_model_name in trained_models:
#             joblib.dump(trained_models[best_model_name], f'models/saved_models/{best_model_name}_model.pkl')
#             joblib.dump(scaler, 'models/saved_models/scaler.pkl')
#             joblib.dump(feature_engineer, 'models/saved_models/feature_engineer.pkl')
#             print(f"\nBest model ({best_model_name}) saved successfully!")
#         elif best_model_name == "Ensemble":
#             joblib.dump(ensemble_model, 'models/saved_models/ensemble_model.pkl')
#             joblib.dump(scaler, 'models/saved_models/scaler.pkl')
#             joblib.dump(feature_engineer, 'models/saved_models/feature_engineer.pkl')
#             print(f"\nEnsemble model saved successfully!")
#     else:
#         print("No models were successfully trained and evaluated.")
    
#     print("\n=== Pipeline Complete ===")

# def quick_preprocess(df, data_processor):
#     """Quick preprocessing without lemmatization"""
#     processed_df = df.copy()
    
#     # Combine title and text if both exist
#     if 'title' in df.columns and 'text' in df.columns:
#         processed_df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
#     elif 'title' in df.columns:
#         processed_df['content'] = df['title'].fillna('')
#     elif 'text' in df.columns:
#         processed_df['content'] = df['text'].fillna('')
#     else:
#         print("No title or text column found!")
#         return None
    
#     # Clean text
#     print("Cleaning text...")
#     processed_df['cleaned_content'] = processed_df['content'].apply(data_processor.clean_text)
    
#     # Remove stopwords
#     print("Removing stopwords...")
#     processed_df['content_no_stopwords'] = processed_df['cleaned_content'].apply(data_processor.remove_stopwords)
    
#     # Skip lemmatization for speed, use content_no_stopwords instead
#     processed_df['lemmatized_content'] = processed_df['content_no_stopwords']
    
#     # Extract sentiment features
#     print("Extracting sentiment features...")
#     sentiment_features = processed_df['cleaned_content'].apply(data_processor.get_sentiment_features)
#     sentiment_df = pd.json_normalize(sentiment_features)
    
#     # Combine with main dataframe
#     processed_df = pd.concat([processed_df, sentiment_df], axis=1)
    
#     return processed_df

# if __name__ == "__main__":
#     main()















import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.models import TraditionalModels
from src.evaluation import ModelEvaluator

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow is available")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - skipping deep learning models")

def create_directories():
    """Create necessary directories"""
    directories = [
        'models/saved_models',
        'models/plots',
        'data/processed'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def quick_preprocess(df, data_processor):
    """Quick preprocessing without lemmatization for faster processing"""
    print("Quick preprocessing...")
    processed_df = df.copy()
    
    # Combine title and text if both exist
    if 'title' in df.columns and 'text' in df.columns:
        processed_df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'title' in df.columns:
        processed_df['content'] = df['title'].fillna('')
    elif 'text' in df.columns:
        processed_df['content'] = df['text'].fillna('')
    else:
        print("❌ No 'title' or 'text' column found!")
        return None
    
    print("Cleaning text...")
    processed_df['cleaned_content'] = processed_df['content'].apply(data_processor.clean_text)
    
    print("Removing stopwords...")
    processed_df['content_no_stopwords'] = processed_df['cleaned_content'].apply(data_processor.remove_stopwords)
    
    # Skip lemmatization for speed
    processed_df['lemmatized_content'] = processed_df['content_no_stopwords']
    
    print("Extracting sentiment features...")
    # Extract sentiment features
    sentiment_features = processed_df['cleaned_content'].apply(data_processor.get_sentiment_features)
    sentiment_df = pd.json_normalize(sentiment_features)
    processed_df = pd.concat([processed_df, sentiment_df], axis=1)
    
    return processed_df

def train_simple_cnn(X_train_text, X_test_text, y_train, y_test, max_features=5000, max_length=100):
    """Train a simple CNN model for text classification"""
    if not TENSORFLOW_AVAILABLE:
        print("⚠️ TensorFlow not available - skipping CNN")
        return None, None
    
    print("Training CNN model...")
    
    # Tokenize text
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)
    
    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, truncating='post')
    
    # Build model
    model = Sequential([
        Embedding(max_features, 128, input_length=max_length),
        Conv1D(64, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    history = model.fit(
        X_train_pad, y_train,
        epochs=3,  # Reduced epochs for speed
        batch_size=64,
        validation_data=(X_test_pad, y_test),
        verbose=1
    )
    
    # Make predictions
    y_pred_proba = model.predict(X_test_pad)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    return model, y_pred

def main():
    """Main execution function"""
    print("=== Optimized Fake News Detection Pipeline ===")
    
    # Create directories
    create_directories()
    
    # Initialize components
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    traditional_models = TraditionalModels()
    evaluator = ModelEvaluator()
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = data_processor.load_data('data/raw/Fake.csv', 'data/raw/True.csv')
    
    if df is None:
        print("❌ Failed to load data!")
        return
    
    print(f"Fake news articles: {len(df[df['label'] == 1])}")
    print(f"True news articles: {len(df[df['label'] == 0])}")
    print(f"Total articles: {len(df)}")
    
    # Sample data for faster processing
    print("\nSampling data for faster processing...")
    df_sampled = df.groupby('label').apply(lambda x: x.sample(min(5000, len(x)), random_state=42)).reset_index(drop=True)
    print(f"Sampled dataset shape: {df_sampled.shape}")
    print("Label distribution:")
    print(df_sampled['label'].value_counts())
    
    # Quick preprocessing
    processed_df = quick_preprocess(df_sampled, data_processor)
    if processed_df is None:
        return
    
    # Step 2: Feature engineering
    print("\n2. Feature engineering...")
    features_df = feature_engineer.extract_text_features(processed_df)
    
    # Step 3: Prepare data for modeling
    print("\n3. Preparing data for modeling...")
    
    # Select important features only
    feature_columns = [
        'char_count', 'word_count', 'sentence_count', 'avg_word_length',
        'uppercase_ratio', 'punctuation_count', 'exclamation_count', 
        'question_count', 'textblob_polarity', 'textblob_subjectivity',
        'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral'
    ]
    
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
    
    # Prepare text data for TF-IDF
    train_texts = features_df.loc[X_train_feat.index, 'content_no_stopwords'].fillna('')
    test_texts = features_df.loc[X_test_feat.index, 'content_no_stopwords'].fillna('')
    
    print("Creating TF-IDF features...")
    # Create TF-IDF features with reduced vocabulary
    tfidf_train = feature_engineer.create_tfidf_features(train_texts, max_features=3000)
    tfidf_test = feature_engineer.tfidf_vectorizer.transform(test_texts)
    
    # Combine features
    X_train_combined = np.hstack([X_train_feat_scaled, tfidf_train.toarray()])
    X_test_combined = np.hstack([X_test_feat_scaled, tfidf_test.toarray()])
    
    print(f"Combined feature shape: {X_train_combined.shape}")
    
    # Step 4: Train traditional ML models
    print("\n4. Training traditional ML models...")
    
    # Define models to train
    models_to_train = [
        'logistic_regression',
        'random_forest', 
        'gradient_boosting',
        'svm',
        'gaussian_nb',
        'xgboost',
        'lightgbm',
        'catboost'
    ]
    
    trained_models = {}
    
    for model_name in models_to_train:
        try:
            print(f"Training {model_name}...")
            model = traditional_models.train_model(model_name, X_train_combined, y_train)
            if model is not None:
                trained_models[model_name] = model
                print(f"✓ {model_name} trained successfully")
            else:
                print(f"✗ {model_name} training failed")
        except Exception as e:
            print(f"✗ Error training {model_name}: {e}")
    
    # Step 5: Evaluate traditional models
    print("\n5. Evaluating traditional models...")
    
    for model_name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test_combined)
            y_pred_proba = model.predict_proba(X_test_combined)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            evaluator.add_result(model_name, accuracy, f1, auc, y_test, y_pred)
            print(f"{model_name}: Accuracy = {accuracy:.4f}, F1-Score = {f1:.4f}, AUC = {auc:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Step 6: Train ensemble model
    print("\n6. Training ensemble model...")
    try:
        # Select top 3 models for ensemble
        if len(trained_models) >= 3:
            ensemble_models = list(trained_models.values())[:3]
            ensemble_predictions = []
            
            for model in ensemble_models:
                pred_proba = model.predict_proba(X_test_combined)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_combined)
                ensemble_predictions.append(pred_proba)
            
            # Average predictions
            ensemble_pred_proba = np.mean(ensemble_predictions, axis=0)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
            
            # Evaluate ensemble
            accuracy = accuracy_score(y_test, ensemble_pred)
            f1 = f1_score(y_test, ensemble_pred)
            auc = roc_auc_score(y_test, ensemble_pred_proba)
            
            evaluator.add_result("Ensemble", accuracy, f1, auc, y_test, ensemble_pred)
            print(f"Ensemble: Accuracy = {accuracy:.4f}, F1-Score = {f1:.4f}, AUC = {auc:.4f}")
            
            # Store ensemble model
            ensemble_model = {
                'models': ensemble_models,
                'model_names': list(trained_models.keys())[:3]
            }
        else:
            ensemble_model = None
            print("Not enough models for ensemble")
            
    except Exception as e:
        print(f"Error creating ensemble: {e}")
        ensemble_model = None
    
    # Step 7: Train simple deep learning model
    print("\n7. Training simple deep learning model...")
    try:
        cnn_model, cnn_pred = train_simple_cnn(
            train_texts.tolist(), 
            test_texts.tolist(), 
            y_train, 
            y_test,
            max_features=5000,
            max_length=100
        )
        
        if cnn_pred is not None:
            accuracy = accuracy_score(y_test, cnn_pred)
            f1 = f1_score(y_test, cnn_pred)
            auc = roc_auc_score(y_test, cnn_pred)
            
            evaluator.add_result("CNN", accuracy, f1, auc, y_test, cnn_pred)
            print(f"CNN: Accuracy = {accuracy:.4f}, F1-Score = {f1:.4f}, AUC = {auc:.4f}")
        
    except Exception as e:
        print(f"Error training CNN: {e}")
        cnn_model = None
    
    # Step 8: Model comparison and save best models
    print("\n8. Model comparison and visualization...")
    
    # Display results table
    results_df = evaluator.compare_models()
    if results_df is not None and len(results_df) > 0:
        print("\nModel Comparison Results:")
        print(results_df.to_string(index=False))
        
        # Plot comparisons
        try:
            evaluator.plot_model_comparison()
            plt.savefig('models/plots/model_comparison.png', dpi=300, bbox_inches='tight')
            print("📊 Model comparison plot saved to models/plots/model_comparison.png")
        except Exception as e:
            print(f"Error plotting comparison: {e}")
        
        # Get best model
        best_model_name = results_df.iloc[0]['Model']
        print(f"\n🏆 Best performing model: {best_model_name}")
        
        # Save models and preprocessors
        print("\n💾 Saving models and preprocessors...")
        try:
            # Save the best traditional model
            if best_model_name in trained_models:
                joblib.dump(trained_models[best_model_name], 'models/saved_models/best_model.pkl')
                joblib.dump(trained_models[best_model_name], f'models/saved_models/{best_model_name}_model.pkl')
                print(f"✅ {best_model_name} model saved!")
            elif best_model_name == "Ensemble" and ensemble_model is not None:
                joblib.dump(ensemble_model, 'models/saved_models/ensemble_model.pkl')
                joblib.dump(ensemble_model, 'models/saved_models/best_model.pkl')
                print(f"✅ Ensemble model saved!")
            
            # Always save LightGBM if available (for web app compatibility)
            if 'lightgbm' in trained_models:
                joblib.dump(trained_models['lightgbm'], 'models/saved_models/lightgbm_model.pkl')
                print("✅ LightGBM model saved!")
            
            # Save preprocessors
            joblib.dump(scaler, 'models/saved_models/scaler.pkl')
            joblib.dump(feature_engineer, 'models/saved_models/feature_engineer.pkl')
            print("✅ Scaler and feature engineer saved!")
            
            # Save CNN model if available
            if TENSORFLOW_AVAILABLE and cnn_model is not None:
                cnn_model.save('models/saved_models/cnn_model.h5')
                print("✅ CNN model saved!")
            
            # Save model metadata
            model_info = {
                'best_model': best_model_name,
                'feature_columns': feature_columns,
                'model_performance': results_df.to_dict('records'),
                'training_samples': len(df_sampled),
                'total_features': X_train_combined.shape[1]
            }
            
            import json
            with open('models/saved_models/model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            print("✅ Model metadata saved!")
            
            print(f"\n📁 All models saved in: models/saved_models/")
            print("📋 Files saved:")
            print("   - best_model.pkl (best performing model)")
            print("   - lightgbm_model.pkl (for web app)")
            print("   - scaler.pkl (feature scaler)")
            print("   - feature_engineer.pkl (feature engineering pipeline)")
            print("   - model_info.json (model metadata)")
            if TENSORFLOW_AVAILABLE and cnn_model is not None:
                print("   - cnn_model.h5 (CNN model)")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    else:
        print("❌ No models were successfully trained and evaluated.")
    
    print("\n=== Pipeline Complete ===")
    print("🚀 You can now run the web application:")
    print("   python app.py")
    print("   or")
    print("   python run_app.py")

if __name__ == "__main__":
    main()
