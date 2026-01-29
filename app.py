
import streamlit as st
import nltk
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# ----------------------------
# SAFE NLTK DOWNLOADS (Cloud Safe)
# ----------------------------
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt")
    nltk.download("punkt_tab")   # REQUIRED for Python 3.12+
    nltk.download("stopwords")

download_nltk_resources()

stop_words = set(stopwords.words("english"))

# ----------------------------
# Load CSS
# ----------------------------
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS optional

load_css("style.css")

# ----------------------------
# SAFE SENTENCE TOKENIZER
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
    text = re.sub(r"\s+", " ", text)

    sentences = safe_sent_tokenize(text)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    cleaned_sentences = []
    for sentence in sentences:
        words = [
            word for word in sentence
            if word.isalpha() and word not in stop_words
        ]
        if words:
            cleaned_sentences.append(words)

    return cleaned_sentences

# ----------------------------
# Streamlit UI
# ----------------------------
st.markdown("<h1>🔍 NLP-Word2Vec-Similarity-Finder-WebApp</h1>", unsafe_allow_html=True)
st.markdown(
    "<p>Enter text and a word to find semantic similarity using Word2Vec</p>",
    unsafe_allow_html=True,
)

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
# Action Button
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
                workers=4
            )

            word = target_word.lower()

            if word in model.wv:
                similar_words = model.wv.most_similar(word, topn=top_n)

                st.markdown("### ✅ Similar Words")
                for w, score in similar_words:
                    st.write(f"**{w}** → similarity: `{score:.4f}`")
            else:
                st.error("❌ Word not found in the given text vocabulary")


