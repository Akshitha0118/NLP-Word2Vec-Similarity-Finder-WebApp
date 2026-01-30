import streamlit as st
import nltk
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# ----------------------------
# SAFE NLTK DOWNLOADS (Streamlit Cloud Safe)
# ----------------------------
@st.cache_data
def download_nltk_resources():
    nltk.download("punkt")
    nltk.download("stopwords")

download_nltk_resources()

stop_words = set(stopwords.words("english"))

# ----------------------------
# Load CSS (Optional)
# ----------------------------
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

# ----------------------------
# Safe Sentence Tokenizer
# ----------------------------
def safe_sent_tokenize(text):
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        return text.split(".")

# ----------------------------
# Text Preprocessing
# ----------------------------
def preprocess_text(text):
    text = re.sub(r"\[[0-9]*\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    text = re.sub(r"\d", " ", text)

    sentences = safe_sent_tokenize(text)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    cleaned = []
    for sentence in sentences:
        words = [
            word for word in sentence
            if word.isalpha() and word not in stop_words
        ]
        if words:
            cleaned.append(words)

    return cleaned

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🔍 NLP-Word2Vec-Similarity-Finder-WebApp")

user_text = st.text_area(
    "Enter Text / Paragraph",
    height=200,
    placeholder="Paste or type your text here..."
)

target_word = st.text_input(
    "Enter a word to find similar words",
    placeholder="e.g. freedom"
)

top_n = st.slider("Number of similar words", 1, 10, 5)

# ----------------------------
# Action
# ----------------------------
if st.button("Find Similar Words"):
    if not user_text.strip() or not target_word.strip():
        st.warning("⚠️ Please enter both text and a word")
    else:
        sentences = preprocess_text(user_text)

        if len(sentences) < 1:
            st.error("❌ Not enough valid text to train Word2Vec")
        else:
            model = Word2Vec(
                sentences,
                vector_size=100,
                window=5,
                min_count=1,
                workers=1  # 🔴 IMPORTANT: Cloud-safe
            )

            word = target_word.lower()

            if word in model.wv:
                st.subheader("✅ Similar Words")
                for w, score in model.wv.most_similar(word, topn=top_n):
                    st.write(f"**{w}** → `{score:.4f}`")
            else:
                st.error("❌ Word not found in the given text")

