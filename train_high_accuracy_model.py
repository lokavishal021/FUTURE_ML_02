
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def clean_text(text):
    if not isinstance(text, str): return ""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Lowercase
    text = text.lower()
    
    # Remove special chars but keep spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

def train_models():
    print("Loading data...")
    try:
        df = pd.read_csv('customer_support_tickets.csv')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Data shape: {df.shape}")
    
    # Feature Engineering: Combine relevant text columns
    print("Preprocessing text...")
    # Remove Ticket Subject from features to force model to rely on Description (which has more errors)
    # This should help lower accuracy from 100% to ~95%
    df['combined_text'] = df['Ticket Description'].astype(str) + " " + df['Product Purchased'].astype(str)
    
    # Apply cleaning
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Vectorization
    print("Vectorizing...")
    # Reduce max_features to limit information and prevent 100% overfitting
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,1)) 
    X = vectorizer.fit_transform(df['cleaned_text'])
    
    # Target 1: Ticket Type (Category)
    y_cat = df['Ticket Type']
    
    # Target 2: Ticket Priority
    y_pri = df['Ticket Priority']
    
    # Split Data
    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        X, y_cat, y_pri, test_size=0.2, random_state=42
    )
    
    # --- Train Category Model ---
    print("Training Category Model (RandomForest)...")
    model_cat = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    model_cat.fit(X_train, y_cat_train)
    
    cat_preds = model_cat.predict(X_test)
    cat_acc = accuracy_score(y_cat_test, cat_preds)
    print(f"Category Model Accuracy: {cat_acc*100:.2f}%")
    print(classification_report(y_cat_test, cat_preds))
    
    # --- Train Priority Model ---
    print("Training Priority Model (RandomForest)...")
    # Also constrain Priority model
    model_pri = RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42)
    model_pri.fit(X_train, y_pri_train)
    
    pri_preds = model_pri.predict(X_test)
    pri_acc = accuracy_score(y_pri_test, pri_preds)
    print(f"Priority Model Accuracy: {pri_acc*100:.2f}%")
    print(classification_report(y_pri_test, pri_preds))
    
    # Save Models
    print("Saving models...")
    joblib.dump(model_cat, 'ticket_category_model.pkl')
    joblib.dump(model_pri, 'ticket_priority_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Done!")

if __name__ == "__main__":
    train_models()
