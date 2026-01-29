import nltk
import re

from gensim.models import Word2Vec
from nltk.corpus import stopwords

# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
the world have come and invaded us, captured our lands, conquered our minds. 
From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
the French, the Dutch, all of them came and looted us, took over what was ours. 
Yet we have not done this to any other nation. We have not conquered anyone. 
We have not grabbed their land, their culture, their history and tried to enforce our way of life on them. 
Why? Because we respect the freedom of others. That is why my first vision is that of freedom.
"""

# ----------------------------
# Text preprocessing
# ----------------------------
text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
text = re.sub(r'\s+', ' ', text)
text = text.lower()
text = re.sub(r'\d', ' ', text)
text = re.sub(r'\s+', ' ', text)

# ----------------------------
# Sentence & word tokenization
# ----------------------------
sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

stop_words = set(stopwords.words('english'))

for i in range(len(sentences)):
    sentences[i] = [
        word for word in sentences[i]
        if word not in stop_words and word.isalpha()
    ]

# ----------------------------
# Train Word2Vec model
# ----------------------------
model = Word2Vec(
    sentences,
    min_count=1,
)

similar = model.wv.most_similar('freedom')

words_only = [word for word, score in similar]

print("Most similar words to 'freedom':", words_only)

import pickle

with open("word2vec_model.pkl", "wb") as file:
    pickle.dump(model, file)
    
import os
os.getcwd()
