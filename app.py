import streamlit as st
import pickle
import os

# ----------------------------
# Load Word2Vec model
# ----------------------------
model_path = 'word2vec_model.pkl'

with open(model_path, "rb") as file:
    model = pickle.load(file)

# ----------------------------
# Load CSS
# ----------------------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ----------------------------
# Streamlit UI
# ----------------------------
st.markdown("<h1>🔍 Word Similarity Finder (Word2Vec)</h1>", unsafe_allow_html=True)
st.markdown("<p>Type a word and get similar words using Word2Vec</p>", unsafe_allow_html=True)

user_input = st.text_input("Enter a word:", placeholder="e.g. freedom")

top_n = st.slider("Number of similar words", 1, 10, 5)

if st.button("Find Similar Words"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a word")
    else:
        word = user_input.lower()
        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=top_n)

            st.markdown("### ✅ Similar Words")
            for w, score in similar_words:
                st.write(f"**{w}**  → similarity: `{score:.4f}`")
        else:
            st.error("❌ Word not found in vocabulary")

