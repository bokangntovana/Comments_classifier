# src/model_training.py

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
import numpy as np
import tabulate
from time import time

def create_pipelines(random_state=42):
    """
    Defines the machine learning pipelines (TFIDF + Classifier).
    Uses best hyperparameters found during notebook exploration.
    """
    
    # 1. Logistic Regression Pipeline (best C=100 from original tuning)
    pipeline_logit = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer="word")),
        ('classifier', LogisticRegression(C=100, max_iter=2000, random_state=random_state))
    ])

    # 2. Random Forest Pipeline (best criterion='entropy' from original tuning)
    pipeline_forest = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer="word")),
        ('classifier', RandomForestClassifier(criterion='entropy', random_state=random_state))
    ])

    # 3. Multinomial Naive Bayes Pipeline (best alpha=0.5, fit_prior=False from original tuning)
    pipeline_mnb = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer="word")),
        ('classifier', MultinomialNB(alpha=0.5, fit_prior=False))
    ])
    
    return {
        "LogisticRegression": pipeline_logit,
        "RandomForest": pipeline_forest,
        "MultinomialNB": pipeline_mnb
    }

def evaluate_models(pipelines, X_train, y_train, X_test, y_test, cv=5):
    """
    Performs k-fold cross-validation on the training set and 
    calculates final F1 scores on the separate test set for comparison.
    """
    
    print("\n" + "="*80)
    print("--- Model Evaluation (Cross-Validation) ---")
    print("="*80)
    
    # --- Cross-Validation ---
    cv_metrics = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    for name, clf in pipelines.items():
        now = time()
        
        scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=cv, n_jobs=8)
        
        # Calculate mean and 95% confidence interval (STD * 2)
        mean_accuracy = np.mean(scores['test_accuracy'])
        std_accuracy = np.std(scores['test_accuracy']) * 2
        
        mean_precision = np.mean(scores['test_precision_weighted'])
        std_precision = np.std(scores['test_precision_weighted']) * 2
        
        mean_recall = np.mean(scores['test_recall_weighted'])
        std_recall = np.std(scores['test_recall_weighted']) * 2
        
        mean_f1 = np.mean(scores['test_f1_weighted'])
        std_f1 = np.std(scores['test_f1_weighted']) * 2
        
        cv_metrics.append({
            'model': name,
            'fit_time': f"{np.sum(scores['fit_time']):.2f}s",
            'accuracy': f"{mean_accuracy:.5f} (STD +/- {std_accuracy:.2f})",
            'precision': f"{mean_precision:.5f} (STD +/- {std_precision:.2f})",
            'recall': f"{mean_recall:.5f} (STD +/- {std_recall:.2f})",
            'f1': f"{mean_f1:.5f} (STD +/- {std_f1:.2f})"
        })
        
        print(f"--- {name} CV completed in {time() - now:.2f}s ---")

    # --- FIX APPLIED HERE ---
    # Use 'keys' to automatically extract headers from the list of dictionaries.
    print(tabulate.tabulate(cv_metrics, headers='keys', tablefmt="fancy_grid"))

    # --- Test Set Evaluation ---
    print("\n" + "="*80)
    print("--- Test Set F1 Scores ---")
    print("="*80)
    
    test_f1_scores = {}
    
    # Fit the models on the full training set before predicting on the test set
    for name, clf in pipelines.items():
        # NOTE: Fit time is included in the CV stats, but we refit here for test prediction.
        clf.fit(X_train, y_train) 
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        test_f1_scores[name] = f1
        
    return test_f1_scores
    
def predict_new_texts(model, texts, target_names):
    """Uses a trained model to predict probabilities for new text samples."""
    
    new_results = model.predict_proba(texts)
    
    print("\n--- Example Predictions ---")
    for i, sentence in enumerate(texts):
        # Determine the predicted class for cleaner output
        predicted_class_index = np.argmax(new_results[i])
        predicted_label = target_names[predicted_class_index]
        
        print(f"\nText: [{sentence}]")
        print(f"Prediction: {predicted_label}")
        print("Probabilities:")
        print(f"  - {target_names[0]} (non-toxic): {new_results[i][0]:.4f}")
        print(f"  - {target_names[1]} (toxic): {new_results[i][1]:.4f}")