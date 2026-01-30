import streamlit as st
import re
import nltk

from nltk.corpus import stopwords
from gensim.models import Word2Vec

# ----------------------------
# NLTK SETUP (ABSOLUTE SAFE)
# ----------------------------
@st.cache_data(show_spinner=False)
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

setup_nltk()
stop_words = set(stopwords.words("english"))

# ----------------------------
# SIMPLE TOKENIZER (NO CRASH)
# ----------------------------
def tokenize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    return [w for w in words if w not in stop_words]

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🔍 Word2Vec Similarity Finder")

text = st.text_area("Enter text", height=200)
word = st.text_input("Target word")
top_n = st.slider("Top N", 1, 10, 5)

# ----------------------------
# ACTION
# ----------------------------
if st.button("Find Similar Words"):
    if not text or not word:
        st.warning("Please enter text and a word")
    else:
        tokens = tokenize_text(text)

        if len(tokens) < 5:
            st.error("Not enough valid words")
        else:
            model = Word2Vec(
                [tokens],          # 👈 IMPORTANT
                vector_size=100,
                window=5,
                min_count=1,
                workers=1          # 👈 Cloud safe
            )

            if word.lower() in model.wv:
                st.success("Similar Words")
                for w, s in model.wv.most_similar(word.lower(), topn=top_n):
                    st.write(f"**{w}** → `{s:.4f}`")
            else:
                st.error("Word not found in vocabulary")


