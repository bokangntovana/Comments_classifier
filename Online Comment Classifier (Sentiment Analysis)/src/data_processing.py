# src/data_processing.py

import pandas as pd
import numpy as np
import re
import nltk
# NOTE: Ensure you have downloaded the 'stopwords' resource for NLTK 
# by running 'python -c "import nltk; nltk.download('stopwords')"' in your terminal
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

def load_data(url):
    """Loads the toxic comments dataset."""
    try:
        comments = pd.read_csv(url)
    except FileNotFoundError:
        # Raise a custom error with a clear message for the user
        raise FileNotFoundError(f"Dataset not found at {url}. Please ensure your dataset is located in the root 'data/' directory.")

    comments['toxic'] = comments.toxic.astype(int)
    
    # Drop the redundant index column if it exists (as seen in the notebook)
    if 'Unnamed: 0' in comments.columns:
        comments = comments.drop(columns=['Unnamed: 0'])
        
    return comments

def clean_text(raw_text):
    """
    Performs text cleaning for NLP: 
    1. Removes HTML tags.
    2. Removes non-alphabetic characters.
    3. Converts to lowercase and removes English stop words.
    """
    # 1. Remove HTML (using a robust parser)
    review_text = BeautifulSoup(raw_text, 'html.parser').get_text()
    
    # 2. Remove non-letters and convert to lower case
    # This keeps only words and spaces, effectively handling punctuation and numbers.
    letters_only = re.sub("[^a-zA-Z]", " ", review_text).lower()
    
    # 3. Tokenize and remove stop words
    words = letters_only.split()
    stops = set(stopwords.words("english"))
    # Filter out stop words and any resulting empty strings
    meaningful_words = [w for w in words if w not in stops and w != ''] 
    
    return " ".join(meaningful_words)

def preprocess_data(comments_df, balance_classes=False, random_state=42):
    """
    Applies text cleaning and optionally balances classes (via downsampling).
    Returns X (features: clean text) and y (target: toxic label).
    """
    
    # --- Class Balancing (Downsampling) ---
    if balance_classes:
        # Separate classes
        non_toxic_comments = comments_df[comments_df['toxic'] == 0]
        toxic_comments = comments_df[comments_df['toxic'] == 1]
        
        # Sample non-toxic comments to match the minority class size (650 in the original data)
        sampled_non_toxic_comments = non_toxic_comments.sample(
            n=len(toxic_comments), random_state=random_state
        )
        
        # Combine the balanced set and shuffle (frac=1)
        combined_df = pd.concat(
            [sampled_non_toxic_comments, toxic_comments], ignore_index=True
        ).sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        # Use the original, imbalanced data
        combined_df = comments_df.copy()

    # --- Apply Cleaning ---
    combined_df['clean_text'] = combined_df['text'].apply(clean_text)

    # Separate features (X) and target (y)
    X = combined_df['clean_text'].values
    y = combined_df['toxic'].values
    
    return X, y