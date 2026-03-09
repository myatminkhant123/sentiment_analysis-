import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis App with Explainability", page_icon="🤖", layout="wide")

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
    
    # Train-test split
    train_text, test_text, y_train, y_test = train_test_split(df['cleaned_review'], df['label'], test_size=0.2, random_state=42)
    
    # Create Pipeline with Vectorizer and SVM (with probability=True for LIME)
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=3000, stop_words='english'),
        SVC(kernel='linear', probability=True, random_state=42)
    )
    pipeline.fit(train_text, y_train)
    
    # Predictions
    svm_pred = pipeline.predict(test_text)
    accuracy = accuracy_score(y_test, svm_pred)
    cm = confusion_matrix(y_test, svm_pred, labels=pipeline.classes_)
    
    # Initialize LIME Explainer
    explainer = LimeTextExplainer(class_names=pipeline.classes_)
    
    return pipeline, explainer, accuracy, cm, df

# Call the cached function
pipeline, explainer, accuracy, cm, df = initialize_model()

# --- APP UI ---
st.title("🤖 AI Review Insights with Explainability")
st.markdown("This application uses **Support Vector Machines (SVM)** to determine sentiment. Now with **LIME Explainability** to show you *why* the model made its decision!")

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
        
        if st.button("Predict Sentiment & Explain"):
            if user_input.strip() == "":
                st.warning("Please enter some text first.")
            else:
                cleaned_input = clean_text(user_input)
                prediction = pipeline.predict([cleaned_input])[0]
                
                if prediction.lower() == 'positive':
                    st.success(f"**Sentiment Result: {prediction}** 🎉")
                else:
                    st.error(f"**Sentiment Result: {prediction}** 😟")
                    
                st.markdown("### Why did the model predict this?")
                st.markdown("The words highlighted below show which terms influenced the model most (Blue for POSITIVE, Orange for NEGATIVE).")
                
                with st.spinner("Generating explainability report..."):
                    # Generate LIME explanation
                    exp = explainer.explain_instance(cleaned_input, pipeline.predict_proba, num_features=10)
                    html_content = f"<div style='background-color: #ffffff; padding: 20px; border-radius: 10px;'>{exp.as_html()}</div>"
                    components.html(html_content, height=450, scrolling=True)
                    
        st.markdown("---")
        st.subheader("Need an example?")
        st.info('Copy and paste this: "I absolutely love this app, it helps me so much!"')
        st.warning('Or this: "This update is terrible and it keeps crashing."')

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
                        xticklabels=pipeline.classes_, yticklabels=pipeline.classes_, ax=ax2)
            ax2.set_xlabel('Predicted Label')
            ax2.set_ylabel('Actual Label')
            st.pyplot(fig2)
            
        st.subheader("Dataset Preview")
        st.dataframe(df.head(5))
