# main.py

import os
from sklearn.model_selection import train_test_split
# Import all necessary functions from your custom modules
from src.data_processing import load_data, preprocess_data
from src.model_training import create_pipelines, evaluate_models, predict_new_texts

# --- Configuration ---
# NOTE: Ensure your CSV file is in a folder named 'data' in your root directory.
DATA_PATH = os.path.join('data', 'toxic_comments.csv') 
RANDOM_STATE = 42

def main():
    print("--- Starting Toxic Comments Classifier Project ---")

    # 1. Data Loading and Preprocessing (with Class Balancing)
    try:
        # 1a. Load Data
        raw_comments_df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure your dataset is located at: {DATA_PATH}")
        return

    # 1b. Preprocess the data and balance classes by downsampling (as in notebook)
    X, y = preprocess_data(
        raw_comments_df, 
        balance_classes=True, 
        random_state=RANDOM_STATE
    )
    
    print(f"\nProcessed dataset size (after balancing): {X.shape[0]} samples")
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # 3. Model Creation and Evaluation
    # Pipelines are created with the best hyperparameters identified in your notebook.
    pipelines = create_pipelines(random_state=RANDOM_STATE)
    
    # 4. Evaluate and get test F1 scores
    test_f1_scores = evaluate_models(
        pipelines, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        cv=5
    )
    
    # 5. Display Final Comparison
    print("\n--- Final Model Comparison (Test Set F1) ---")
    for model, score in sorted(test_f1_scores.items(), key=lambda item: item[1], reverse=True):
        print(f" {model}: {score:.4f}")

    # 6. Example Prediction
    best_model_name = "LogisticRegression" 
    best_model = pipelines[best_model_name]
    target_names = ['non-toxic', 'toxic']

    new_reviews = [
        "the king is an aweful human and must shut up", 
        "I really enjoyed the flight, but the food was bad"
    ]
    
    predict_new_texts(best_model, new_reviews, target_names)

if __name__ == "__main__":
    # If NLTK stopwords still throws a LookupError, run:
    # python -c "import nltk; nltk.download('stopwords')"
    main()