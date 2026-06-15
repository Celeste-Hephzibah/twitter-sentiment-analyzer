import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

model = pickle.load(open('model/sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="🐦")
st.title("🐦 Twitter Sentiment Analyzer")
st.write("Analyze the sentiment of any tweet or text instantly.")

tab1, tab2 = st.tabs(["Single Tweet", "Bulk Analysis"])

with tab1:
    user_input = st.text_area("Enter a tweet or any text:", height=120,
        placeholder="e.g. Just tried the new phone and it's absolutely amazing!")
    if st.button("Analyze", key="single"):
        if user_input.strip():
            cleaned = clean_tweet(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            st.divider()
            if prediction == 1:
                st.success("✅ Positive Sentiment")
                st.balloons()
            else:
                st.error("❌ Negative Sentiment")
            with st.expander("See what the model processed"):
                st.write(f"**Original:** {user_input}")
                st.write(f"**Cleaned:** {cleaned}")
        else:
            st.warning("Please enter some text first.")

with tab2:
    bulk_input = st.text_area("Paste multiple tweets (one per line):", height=200,
        placeholder="Tweet 1\nTweet 2\nTweet 3")
    if st.button("Analyze All", key="bulk"):
        if bulk_input.strip():
            tweets = [t.strip() for t in bulk_input.split('\n') if t.strip()]
            cleaned_tweets = [clean_tweet(t) for t in tweets]
            vectorized = tfidf.transform(cleaned_tweets)
            predictions = model.predict(vectorized)
            results_df = pd.DataFrame({
                'Tweet': tweets,
                'Sentiment': ['✅ Positive' if p==1 else '❌ Negative' for p in predictions]
            })
            st.dataframe(results_df, use_container_width=True)
            pos_count = sum(predictions)
            neg_count = len(predictions) - pos_count
            fig, ax = plt.subplots()
            ax.bar(['Positive', 'Negative'], [pos_count, neg_count],
                   color=['#2ecc71', '#e74c3c'])
            ax.set_title('Sentiment Distribution')
            st.pyplot(fig)
        else:
            st.warning("Please enter at least one tweet.")