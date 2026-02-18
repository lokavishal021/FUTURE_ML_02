import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix

# Ensure NLTK downloads
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Helper function
def clean_text(text):
    if not isinstance(text, str): return ""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

print("Loading data and models...")
try:
    df = pd.read_csv('customer_support_tickets.csv')
    model_cat = joblib.load('ticket_category_model.pkl')
    model_pri = joblib.load('ticket_priority_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except Exception as e:
    print(f"Error loading: {e}")
    exit(1)

# Preprocess
print("Preprocessing dataset...")
df['full_text'] = df['Ticket Subject'].astype(str) + " " + df['Ticket Description'].astype(str) + " " + df['Product Purchased'].astype(str)
df['cleaned_text'] = df['full_text'].apply(clean_text)

# Vectorize
X = vectorizer.transform(df['cleaned_text']).toarray()
y_cat_true = df['Ticket Type']
y_pri_true = df['Ticket Priority']

# Predict
print("Generating predictions...")
y_cat_pred = model_cat.predict(X)
y_pri_pred = model_pri.predict(X)

# Plot Confusion Matrix - Category
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_cat_true, y_cat_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=model_cat.classes_, yticklabels=model_cat.classes_)
plt.title('Confusion Matrix - Categories')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_category.png')
print("Saved confusion_matrix_category.png")

# Plot Confusion Matrix - Priority
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_pri_true, y_pri_pred), annot=True, fmt='d', cmap='Oranges',
            xticklabels=model_pri.classes_, yticklabels=model_pri.classes_)
plt.title('Confusion Matrix - Priority')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_priority.png')
print("Saved confusion_matrix_priority.png")

print("Done.")
