import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Page Configuration
st.set_page_config(
    page_title="ChatGPT Sentiment Analysis",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for UI polish
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .positive-bg {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .negative-bg {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .neutral-bg {
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
    }
    </style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('CHATGPT.csv')
    df.dropna(subset=['Review', 'label'], inplace=True)
    df['cleaned_review'] = df['Review'].apply(clean_text)
    
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['label']
    
    base_model = LinearSVC(random_state=42)
    calibrated_model = CalibratedClassifierCV(estimator=base_model)
    calibrated_model.fit(X, y)
    
    return calibrated_model, vectorizer, len(df), df['label'].value_counts()

# Load model and dataset stats
try:
    model, vectorizer, total_samples, label_counts = load_and_train_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load dataset or train model: {e}")
    model_loaded = False

# Sidebar Info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/chatgpt.png", width=70)
    st.title("Project Info")
    st.markdown("**ChatGPT Reviews Sentiment Classifier**")
    st.markdown("This NLP model classifies user reviews into **POSITIVE**, **NEGATIVE**, or **NEUTRAL** sentiments.")
    st.divider()
    if model_loaded:
        st.subheader("📊 Dataset Statistics")
        st.metric("Total Reviews Analyzed", f"{total_samples:,}")
        for label, count in label_counts.items():
            st.write(f"- **{label.capitalize()}**: {count:,}")
    st.divider()
    st.caption("Powered by TF-IDF & Support Vector Machine (LinearSVC)")

# Main Header
st.markdown('<div class="main-title">🤖 ChatGPT Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter any review or text feedback below to predict sentiment.</div>', unsafe_allow_html=True)

if model_loaded:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Review")
        
        # Sample buttons
        st.markdown("**Try sample reviews:**")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        sample_text = ""
        if btn_col1.button("👍 Positive Example"):
            sample_text = "ChatGPT has completely transformed my workflow! It saves me hours of coding and writing every day."
        if btn_col2.button("👎 Negative Example"):
            sample_text = "It keeps giving incorrect code syntax and hallucinating fake references. Quite frustrating."
        if btn_col3.button("😐 Neutral Example"):
            sample_text = "I tested ChatGPT today for some basic tasks. It works okay for simple queries."
            
        user_input = st.text_area(
            "Type or paste a review:",
            value=sample_text,
            height=160,
            placeholder="e.g., ChatGPT is extremely helpful for explaining complex concepts..."
        )
        
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
    with col2:
        st.subheader("Prediction Result")
        if analyze_btn or user_input.strip():
            if user_input.strip():
                cleaned = clean_text(user_input)
                vec_input = vectorizer.transform([cleaned])
                prediction = model.predict(vec_input)[0]
                probabilities = model.predict_proba(vec_input)[0]
                classes = model.classes_
                
                # Format prediction result
                pred_upper = str(prediction).upper()
                if "POS" in pred_upper:
                    bg_class = "positive-bg"
                    emoji = "😊"
                elif "NEG" in pred_upper:
                    bg_class = "negative-bg"
                    emoji = "😟"
                else:
                    bg_class = "neutral-bg"
                    emoji = "😐"
                    
                st.markdown(f'<div class="sentiment-box {bg_class}">{emoji} {pred_upper}</div>', unsafe_allow_html=True)
                
                st.markdown("### Confidence Breakdown")
                prob_df = pd.DataFrame({
                    'Sentiment': [c.capitalize() for c in classes],
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)
                
                for idx, row in prob_df.iterrows():
                    st.write(f"**{row['Sentiment']}**: {row['Probability']*100:.1f}%")
                    st.progress(float(row['Probability']))
            else:
                st.warning("Please enter some text to analyze.")
        else:
            st.info("Enter a review on the left and click **Analyze Sentiment**.")
