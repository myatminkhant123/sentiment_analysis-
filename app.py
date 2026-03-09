import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🤖", layout="wide")

# Pre-processing function
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

# Load and train the model (cached so it doesn't re-run on every interaction)
@st.cache_resource(show_spinner="Loading dataset and training model... This takes just a few seconds.")
def initialize_model():
    # Load dataset
    try:
        df = pd.read_csv('CHATGPT.csv')
    except Exception as e:
        return None, None, None, None, None
        
    df.dropna(subset=['Review', 'label'], inplace=True)
    df['cleaned_review'] = df['Review'].apply(clean_text)
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train SVM Model
    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Predictions
    svm_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, svm_pred)
    cm = confusion_matrix(y_test, svm_pred, labels=svm_model.classes_)
    
    return svm_model, vectorizer, accuracy, cm, df

# Call the cached function
svm_model, vectorizer, accuracy, cm, df = initialize_model()

# --- APP UI ---
st.title("🤖 ChatGPT Review Sentiment Analyzer")
st.markdown("This application uses natural language processing and **Support Vector Machines (SVM)** to determine if a review is **POSITIVE** or **NEGATIVE**.")

if df is None:
    st.error("Dataset 'CHATGPT.csv' not found. Please make sure it's in the same folder.")
else:
    # Sidebar for Model Insights
    st.sidebar.header("📊 Model Performance")
    st.sidebar.markdown(f"**Accuracy:** `{accuracy:.2%}`")
    st.sidebar.markdown(f"**Dataset Size:** `{len(df)} reviews`")
    
    # Main Tabs
    tab1, tab2 = st.tabs(["💬 Try it Out", "📈 Data & Statistics"])
    
    with tab1:
        st.subheader("Analyze Your Own Text")
        user_input = st.text_area("Enter a review or sentence about ChatGPT:", height=150, placeholder="e.g. This app is incredibly helpful, but history is annoying to access.")
        
        if st.button("Predict Sentiment"):
            if user_input.strip() == "":
                st.warning("Please enter some text first.")
            else:
                cleaned_input = clean_text(user_input)
                vec_input = vectorizer.transform([cleaned_input])
                prediction = svm_model.predict(vec_input)[0]
                
                if prediction.lower() == 'positive':
                    st.success(f"**Sentiment Result: {prediction}** 🎉")
                else:
                    st.error(f"**Sentiment Result: {prediction}** 😟")
                    
        st.markdown("---")
        st.subheader("Or Try an Example:")
        example = "I absolutely love this app, it helps me so much!"
        st.info(f'"{example}" **→ POSITIVE**')
        example2 = "This update is terrible and it keeps crashing."
        st.warning(f'"{example2}" **→ NEGATIVE**')

    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.countplot(x='label', data=df, palette='viridis', ax=ax1)
            ax1.set_title("Positive vs Negative Reviews")
            st.pyplot(fig1)

        with col2:
            st.subheader("Confusion Matrix")
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=svm_model.classes_, yticklabels=svm_model.classes_, ax=ax2)
            ax2.set_xlabel('Predicted Label')
            ax2.set_ylabel('Actual Label')
            st.pyplot(fig2)
            
        st.subheader("Dataset Preview")
        st.dataframe(df.head(5))
