import pandas as pd
import numpy as np
import string
import nltk
import sklearn.feature_extraction.text as sklearn_text
import sklearn.linear_model as sklearn_linear
import sklearn.model_selection as sklearn_model_selection
import sklearn.metrics as sklearn_metrics


X = ["Hello there.<br /><br />",
     "General Kenobi.",
     "Attack, Kenobi!"]

lemmatizer=nltk.stem.WordNetLemmatizer()

for x_row in range(len(X)):
    X[x_row] = X[x_row].replace("<br />", " ")
    X[x_row] = X[x_row].lower()
    X[x_row] = X[x_row].translate(str.maketrans("", "", string.punctuation))
    # stopwords
    X[x_row] = nltk.word_tokenize(X[x_row])
    for x_word in range(len(X[x_row])):
        X[x_row][x_word] = lemmatizer.lemmatize(X[x_row][x_word])
    X[x_row] = " ".join(X[x_row])
    if x_row % 100 == 0:
        print("{}/{} reviews prepared.".format(x_row, len(X)))
else:
    print("{}/{} reviews prepared.".format(len(X), len(X)))

print(X)

print("Vectorisation starting.")
tfidf = sklearn_text.TfidfVectorizer(ngram_range=(1, 2))
X = tfidf.fit_transform(X)
print("Vectorisation finished.")

print(X)

X = tfidf.inverse_transform(0.5773502691896257)

print(X)
