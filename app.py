import streamlit as st
import joblib
import os

# Set page title
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Title
st.title("📰 Fake News Detection App")
st.markdown("Paste a news article or headline below to find out whether it's **Real** or **Fake**.")

# Check if model and vectorizer files exist
model_file = "fake_news_model.pkl"
vectorizer_file = "tfidf_vectorizer.pkl"

if not os.path.exists(model_file) or not os.path.exists(vectorizer_file):
    st.error("❌ Required model files not found. Please make sure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder as this script.")
    st.stop()

# Load model and vectorizer
model = joblib.load(model_file)
vectorizer = joblib.load(vectorizer_file)

# Text input
user_input = st.text_area("🧾 Enter News Text", height=200, placeholder="e.g. Scientists discover water on Mars...")

# Predict button
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text first.")
    else:
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.success("✅ This is *Real News*")
        else:
            st.error("❌ This is *Fake News*")
