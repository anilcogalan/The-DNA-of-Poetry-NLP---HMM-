import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import unidecode
import re
import nltk
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
from sklearn.model_selection import train_test_split
from classifier import Classifier
from helpers import df_col_drop,preprocess_text,text_to_int, initialize_and_compute_matrices,compute_and_normalize_counts

# nltk.download('stopwords')

df_cy = pd.read_json("data/poems_cy.json")
df_nh = pd.read_json("data/poems_nh.json")
df_cy = df_col_drop(df_cy,'link')
df_nh= df_col_drop(df_nh,'link')

df_cy['labels'] = 0
df_nh['labels'] = 1

df = pd.concat([df_cy,df_nh], ignore_index=True)

df['poem_text'] = df['poem_text'].apply(preprocess_text)

poems = df['poem_text']
labels = df['labels']

X_train, X_test, y_train, y_test = train_test_split(poems, labels,test_size=.2)

idx = 1
word2idx = {'<unk>' : 0}

for text in X_train:
    tokens = text.split()
    for token in tokens:
        if token not in word2idx:
            word2idx[token] = idx
            idx += 1

X_train_int = [text_to_int(text, word2idx) for text in X_train]
X_test_int = [text_to_int(text, word2idx) for text in X_test]

A0, pi0, A1, pi1 = initialize_and_compute_matrices(X_train, y_train, word2idx)

logA0, logpi0 = compute_and_normalize_counts([t for t, y in zip(X_train, y_train) if y == 0],word2idx,token, A0, pi0)
logA1, logpi1 = compute_and_normalize_counts([t for t, y in zip(X_train, y_train) if y == 1],word2idx,token, A1, pi1)

count0 = sum(y_train == 0)
count1 = sum(y_train == 1)
total = len(y_train)
logp0 = np.log(count0 / total)
logp1 = np.log(count1 / total)


clf = Classifier([logA0, logA1], [logpi0, logpi1], [logp0, logp1])

Ptrain = clf.predict(X_train_int)
print(f"Train accuracy: {np.mean(Ptrain == y_train)}")

Ptest = clf.predict(X_test_int)
print(f"Test accuracy: {np.mean(Ptest == y_test)}")


