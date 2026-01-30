# NLP-Word2Vec-Similarity-Finder-WebApp

A simple and cloud-safe **Word2Vec Similarity Finder** built using **Streamlit**, **NLTK**, and **Gensim**.  
This app allows users to input text, choose a target word, and find the most similar words using Word2Vec embeddings.

---

## 📌 Features

- Clean and minimal Streamlit UI
- Safe NLTK setup for Streamlit Cloud
- Custom tokenizer (no crashes, no heavy downloads)
- Word similarity using **Gensim Word2Vec**
- Adjustable number of similar words (Top-N)

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit**
- **NLTK**
- **Gensim**
- **Regular Expressions (re)**

---

## 🧠 How It Works

User enters a block of text

Text is cleaned and tokenized

Stopwords are removed using NLTK

A Word2Vec model is trained on the input text

Similar words are generated based on vector similarity

## ⚠️ Notes

Minimum of 5 valid words required to train Word2Vec

Target word must exist in the vocabulary

Model is trained on-the-fly for simplicity

## 📄 requirements.txt
streamlit
nltk
gensim

## 🚀 Future Improvements

Pre-trained Word2Vec / FastText support

Sentence similarity

Multilingual support

Save & reuse trained models

## 👤 Author

Akshitha Hirakari
📍 Banglore 
💡 NLP | Machine Learning | Streamlit Apps

