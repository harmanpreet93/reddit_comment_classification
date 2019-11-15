# -*- coding: utf-8 -*-

"""# Initialize globals"""

import pandas as pd
import numpy as np
import re

TRAIN_DATA_PATH = "data_train.pkl"
TEST_DATA_PATH = "data_test.pkl"

"""# Import the text and classes"""

from sklearn import preprocessing

train_data = pd.read_pickle(TRAIN_DATA_PATH)
test_data  = pd.read_pickle(TEST_DATA_PATH)

nb_X_Train = len(train_data[0])
All_X = np.concatenate((np.array(train_data[0]),np.array(test_data)))

le = preprocessing.LabelEncoder()
y = le.fit_transform(np.array(train_data[1]))

"""# Pre-process the data

Remove stop words and stem
"""
from nltk import download
from nltk.corpus import stopwords
download('stopwords') # nltk

stop_words_list = stopwords.words('english')
pattern = re.compile(r'\b\w\w+\b')
word_count = np.zeros(All_X.shape[0])

for idx, sentence in enumerate(All_X):
  All_X[idx] = " ".join([word for word in re.findall(pattern, sentence.lower()) if word not in stop_words_list])
  word_count[idx] = len(All_X[idx])

"""Count and weight the terms"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_transformer = TfidfVectorizer(
  ngram_range=(1, 1),
  min_df=2,
  strip_accents = "unicode",
  sublinear_tf = True
)
All_X_ifidf = tfidf_transformer.fit_transform(All_X)

X = All_X_ifidf[:nb_X_Train,:]
Kaggle_Test_X = All_X_ifidf[nb_X_Train:,:]

"""# Train the classifier"""

from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

clf_vote =   VotingClassifier(
  estimators=[
    ('sgd',SGDClassifier(**{'penalty': 'l2', 'loss': 'modified_huber', 'class_weight': 'balanced', 'alpha': 0.0001})),
    ('mnb', MultinomialNB(alpha=0.25)),
    ('nn', MLPClassifier(**{'max_iter': 1, 'hidden_layer_sizes': (256,), 'batch_size': 64})),
  ],
  voting='soft',
  weights=[1,1,1],
  n_jobs=-1
)

clf_vote.fit(X, y)

"""# Predict on Kaggle set"""

Kaggle_y_pred = clf_vote.predict(Kaggle_Test_X)
Kaggle_y_pred = le.inverse_transform(Kaggle_y_pred)

ids = [i for i in range(len(Kaggle_y_pred))]
sub_df = pd.DataFrame(data=list(zip(ids, Kaggle_y_pred)), columns=["Id","Category"])
sub_df.to_csv("submission.csv", index=False)
