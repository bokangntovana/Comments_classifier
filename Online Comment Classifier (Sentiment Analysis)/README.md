# Toxic Comment Classifier

## Project Overview
This repository hosts a solution for **Abuse Detection** in online comments, a critical task for maintaining healthy digital platforms. The project's goal is to classify text utterances into one of two categories: **Toxic** (abusive behavior, hate speech, offensive language, sexism, racism) or **Non-Toxic**.

This project was developed as Assignment 2 for a Business Intelligence module, focusing on applying machine learning pipelines and text processing techniques (Natural Language Processing or NLP) to a complex classification problem.

---

## Key Features and Technologies

### Classification Pipeline
The primary solution uses a machine learning pipeline composed of **TF-IDF Vectorization** for feature extraction and a **Logistic Regression** model for classification, as it demonstrated the best balance of performance and efficiency.

### Technology Stack
* **Language:** Python
* **Core Libraries:** `scikit-learn`, `pandas`, `numpy`, `nltk`
* **Text Processing:** TF-IDF (Term Frequency-Inverse Document Frequency), Stop Word Removal
* **Evaluation:** Cross-Validation, F1-Score (Weighted), Confusion Matrix

---

##  Performance & Results

The final models were trained and evaluated on a **balanced dataset** (downsampled to 1,300 samples total) to ensure fair performance measurement across both classes.

### Model Comparison (F1 Score on Test Set)

| Model | Avg. F1 Score (Weighted) | Avg. Accuracy | Time to Fit (s) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (Best)** | 0.79417 | 0.79423 | 0.11 |
| Multinomial Naive Bayes | 0.79997 | 0.80000 | 0.10 |
| Random Forest | 0.73303 | 0.73750 | 1.85 |

### Best Model Performance (Logistic Regression)

The Logistic Regression model achieved an **F1-Score of 0.7852** on the independent test set.

| Metric | Non-Toxic (0) | Toxic (1) |
| :--- | :--- | :--- |
| **Precision** | 0.8045 | 0.7638 |
| **Recall** | 0.7810 | 0.7886 |

*(Note: The full classification report and Confusion Matrix are available in the `toxic_comments_classifier.ipynb` notebook.)* 

---

##  Project Structure and Setup

This project uses a modular, professional structure to separate concerns:

### Getting Started

To set up and run this project locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/bokangntovana/Comments_classifier.git](https://github.com/bokangntovana/Comments_classifier.git)
    cd Comments_classifier
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # On Windows PowerShell:
    . .\.venv\Scripts\Activate.ps1
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data:** The text processing requires the NLTK stopwords resource:
    ```bash
    python -c "import nltk; nltk.download('stopwords')"
    ```

5.  **Run the Main Pipeline:**
    ```bash
    # Ensure the toxic_comments.csv file is in the 'data/' folder first.
    python main.py
    ```