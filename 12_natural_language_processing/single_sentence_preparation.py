import pandas as pd
import numpy as np
import string
import nltk
import nltk.corpus as nltk_corpus
import re


imdb_reviews = pd.read_csv("data/IMDB Dataset.csv")

X = imdb_reviews["review"].iloc[4]
print(X, "\n")

# lemmatizer = nltk.stem.WordNetLemmatizer()
# stemmer = nltk.stem.LancasterStemmer()
stemmer = nltk.stem.PorterStemmer()

stop_words = nltk_corpus.stopwords.words("english")

X = re.sub("<.*?>", " ", X)
X = X.lower()
X = X.translate(str.maketrans("", "", string.punctuation))

X = nltk.word_tokenize(X)

# X = [lemmatizer.lemmatize(word) for word in X]
X = [stemmer.stem(word) for word in X]

# X = [word for word in X if word not in stop_words]

X = " ".join(X)

print(X)
