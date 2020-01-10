import pandas as pd
import numpy as np
import string
import nltk
import sklearn.feature_extraction.text as sklearn_text
import sklearn.linear_model as sklearn_linear
import sklearn.model_selection as sklearn_model_selection
import sklearn.metrics as sklearn_metrics


imdb_reviews = pd.read_csv("data/IMDB Dataset.csv")

# imdb_reviews_count = imdb_reviews.groupby(by="sentiment").count()
# print(imdb_reviews_count)

# N = 1000    # len(imdb_reviews_count)
N = len(imdb_reviews)
# X = imdb_reviews["review"]
X = imdb_reviews["review"][:N]   # TODO remove temporary fix
y = imdb_reviews["sentiment"][:N]

y = y.replace({"positive": 1, "negative": 0})
X = np.array(X)

lemmatizer = nltk.stem.WordNetLemmatizer()
# stemmer = nltk.stem.LancasterStemmer()
# stemmer = nltk.stem.PorterStemmer()

for x_row in range(len(X)):
    X[x_row] = X[x_row].replace("<br />", " ")
    X[x_row] = X[x_row].lower()
    X[x_row] = X[x_row].translate(str.maketrans("", "", string.punctuation))
    # stopwords
    X[x_row] = nltk.word_tokenize(X[x_row])
    for x_word in range(len(X[x_row])):
        X[x_row][x_word] = lemmatizer.lemmatize(X[x_row][x_word])
        # X[x_row][x_word] = stemmer.stem(X[x_row][x_word])
    X[x_row] = " ".join(X[x_row])
    if x_row % 100 == 0:
        print("{}/{} reviews prepared.".format(x_row, len(X)))
else:
    print("{}/{} reviews prepared.".format(len(X), len(X)))
# print(X[3])

print("Vectorisation starting.")
tfidf = sklearn_text.TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
X = tfidf.fit_transform(X)
print("Vectorisation finished.")

# print(X[3])

X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(X, y, test_size=0.25, shuffle=True)

print("Logistic Regression training starting.")

logistic_model = sklearn_linear.LogisticRegression(solver="lbfgs", max_iter=1000)
logistic_model.fit(X_train, y_train)

print("Logistic Regression training finished.")

y_predicted = logistic_model.predict(X_test)

print("Accuracy: {:.2f}%".format(sklearn_metrics.accuracy_score(y_test, y_predicted) * 100))
print(sklearn_metrics.confusion_matrix(y_test, y_predicted))

top_phrase_count = 20
tfidf_feature_names = tfidf.get_feature_names()

print("Negative:")
top_negative_phrases_indexes = np.argsort(logistic_model.coef_[0])[:top_phrase_count]
top_negative_phrases = [tfidf_feature_names[z] for z in top_negative_phrases_indexes]
print(top_negative_phrases)

print("Positive:")
top_positive_phrases_indexes = np.argsort(logistic_model.coef_[0])[-top_phrase_count:]
top_positive_phrases = [tfidf_feature_names[z] for z in top_positive_phrases_indexes]
print(top_positive_phrases)
