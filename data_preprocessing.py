import os
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import nltk

# Download stopwords if not already downloaded
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# Define the path to the dataset
data_path = '/home/ubuntu/twenty_newsgroups/20_newsgroups'

# Load the dataset
newsgroups_data = load_files(data_path, encoding='latin1', random_state=42)

# Function to clean text
def clean_text(text):
    # Remove headers, footers, and quotes
    lines = text.split('\n')
    new_text = []
    for line in lines:
        if line.startswith('From:') or line.startswith('Subject:') or line.startswith('Organization:') or line.startswith('Lines:') or line.startswith('Message-ID:') or line.startswith('References:') or line.startswith('NNTP-Posting-Host:') or line.startswith('In article <'):
            continue
        if line.startswith('>') and len(line) > 1:
            continue
        new_text.append(line)
    text = '\n'.join(new_text)

    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning to all documents
cleaned_documents = [clean_text(doc) for doc in newsgroups_data.data]

# Tokenization and TF-IDF vectorization
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=5000)
X = vectorizer.fit_transform(cleaned_documents)

print(f"Shape of TF-IDF matrix: {X.shape}")

# Save the preprocessed data (TF-IDF matrix) and vectorizer for later use
import pickle

with open('/home/ubuntu/twenty_newsgroups/tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('/home/ubuntu/twenty_newsgroups/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("TF-IDF matrix and vectorizer saved.")

# Save a sample of cleaned text for review
with open('/home/ubuntu/twenty_newsgroups/cleaned_sample.txt', 'w') as f:
    f.write(cleaned_documents[0])

print("Sample of cleaned text saved to /home/ubuntu/twenty_newsgroups/cleaned_sample.txt")

