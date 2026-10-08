import pandas as pd
import numpy as np
import re
import string
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

app = Flask(__name__)

# --- 1. Data Processing & Model Training (Runs on Startup) ---
print("Loading data and training model...")

# Global variables for model and vectorizer
model = None
vectorizer = None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    return text

def train_model():
    global model, vectorizer
    try:
        # Load Dataset
        df = pd.read_csv('CHATGPT.csv')
        df.dropna(subset=['Review', 'label'], inplace=True)
        
        # Pre-process
        df['cleaned_review'] = df['Review'].apply(clean_text)
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        X = vectorizer.fit_transform(df['cleaned_review'])
        y = df['label']
        
        # Train SVM (LinearSVC) - usually performed best in the notebook
        model = LinearSVC(random_state=42)
        model.fit(X, y)
        print("Model trained successfully!")
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

# Train immediately
success = train_model()

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model not trained successfully.'}), 500
    
    data = request.json
    review = data.get('review', '')
    
    if not review:
        return jsonify({'error': 'No review provided'}), 400
    
    # Process and Predict
    cleaned_review = clean_text(review)
    vec_review = vectorizer.transform([cleaned_review])
    prediction = model.predict(vec_review)[0]
    
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    if success:
        print("Starting web server on http://127.0.0.1:5000")
        app.run(debug=True, port=5000)
    else:
        print("Could not start server due to model training failure.")
