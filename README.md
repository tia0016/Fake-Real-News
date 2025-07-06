# 📰 Fake News Detection using Machine Learning

This project is a beginner-friendly machine learning model that detects whether a news article is **real or fake** using **natural language processing (NLP)** techniques.

# 🚀 Overview

- **Model:** Logistic Regression
- **Vectorization:** TF-IDF
- **Accuracy Achieved:** ~98.3%
- **Platform:** Google Colab

# 📦 Dataset

Used the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle, which contains:

- `Fake.csv` — Fake news articles
- `True.csv` — Real news articles

# 🧠 How it Works

1. Combine `Fake.csv` and `True.csv` and assign labels (0 = fake, 1 = real)
2. Preprocess the text data (TF-IDF vectorization)
3. Train a Logistic Regression classifier
4. Evaluate model performance
5. Save the model and test it on custom input

 
### 🔗 Try the live app:
[Click here to use it](https://fake-real-news-yzmoetyuwvxcaqjgt2asya.streamlit.app/)

---
# 📁 Project Structure

📦 Fake-Real-News/
├── Fake_and_Real_News.ipynb ← Main notebook
├── Fake.csv ← Fake news articles
├── True.csv ← Real news articles
├── fake_news_model.pkl ← Saved ML model (optional)
├── tfidf_vectorizer.pkl ← Saved TF-IDF vectorizer (optional)
└── README.md ← You're here

# 🛠 How to Run

1. Upload `Fake.csv` and `True.csv` into Google Colab
2. Open and run `Fake_and_Real_News.ipynb`
3. Save model (optional)
4. Test custom news using:
```python
sample = ["Aliens spotted near the White House!"]
vector = vectorizer.transform(sample)
prediction = model.predict(vector)
print("Fake" if prediction[0] == 0 else "Real")
