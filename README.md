# ChatGPT Review Sentiment Analyzer 🤖📈

Welcome to the **ChatGPT Review Sentiment Analyzer** project! This repository contains a powerful, machine-learning-driven web application that predicts whether text or review sentiment is **POSITIVE** or **NEGATIVE**.

Not only does it classify text using state-of-the-art classic NLP techniques, but it also comes packed with a **LIME (Local Interpretable Model-Agnostic Explanations)** module that visually explains *why* the AI made its decision, making the model transparent and easy to understand.

## 🚀 Features

*   **Real-time Sentiment Prediction:** Type or paste any text review and get an instant determination of sentiment (Positive/Negative).
*   **Model Explainability (LIME):** Dive inside the AI's "brain". The app highlights the exact words that influenced its decision, showing you the probability weights for each word.
*   **Interactive UI:** Built using Streamlit, featuring a beautiful and responsive web interface that supports both light and dark modes.
*   **Data Visualization:** The application displays interactive charts (via Matplotlib and Seaborn) showing dataset distributions and the confusion matrix evaluating the model's performance on the test dataset.

## 🛠️ Technology Stack

1.  **Python 3.x:** The core language powering the machine learning logic.
2.  **Streamlit:** For rapidly turning the data script into an interactive web application.
3.  **Scikit-Learn:** Used for creating the TF-IDF Vectorizer and training the Support Vector Machine (SVM) pipeline.
4.  **LIME (Local Interpretable Model-Agnostic Explanations):** Used for advanced model explainability.
5.  **Pandas & NumPy:** For data manipulation and processing.
6.  **Matplotlib & Seaborn:** For producing detailed and aesthetic data visualizations.

## 🧠 How the Model Works

1.  **Preprocessing:** All input text undergoes extensive cleaning—URLs, HTML tags, punctuation, and newlines are stripped, and the text is converted to lowercase.
2.  **Vectorization:** We utilize a **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer limited to the top 3,000 most significant words, excluding English stop words. This converts the actual words into mathematical numbers.
3.  **Classification:** We utilize a Support Vector Classifier (`SVC(kernel='linear', probability=True)`). SVC searches for the optimal hyperplane to divide our text vectors into the POSITIVE or NEGATIVE classes.
4.  **Explainability:** LIME perturbs the input text and sees how the model's probability predictions change, then weights those perturbations to construct an easy-to-read layout of which words heavily pushed the result in either direction.

## 💻 How to Run Locally

If you'd like to test the project natively on your machine:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/myatminkhant123/sentiment_analysis-.git
    cd sentiment_analysis-
    ```
2.  **Install the Required Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit App**
    ```bash
    python -m streamlit run app.py
    ```

*The app will automatically open in your browser at `http://localhost:8501`.*

## 🌐 Deploying to the Cloud

This project is perfectly crafted for seamless free deployment on Streamlit Community Cloud:

1.  Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
2.  Click **"New App"**.
3.  Select this repository (`myatminkhant123/sentiment_analysis-`), `main` branch, and set the main file path to `app.py`.
4.  Click **Deploy**! Wait a few moments while Streamlit installs the dependencies listed in `requirements.txt`.

## 📂 Project Structure

*   `app.py`: The main Streamlit web application with all UI code and ML pipelines.
*   `NLP_PROJECT.py`: The raw Python script showing the development history from standard execution.
*   `CHATGPT.csv`: The dataset containing raw reviews and their associated positive/negative labels.
*   `requirements.txt`: The file dictating to servers which packages to install.
*   `README.md`: This extensive project guide documentation.

---

*Thank you for exploring this project! Let AI be your guide.*