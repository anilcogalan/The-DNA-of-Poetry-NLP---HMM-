import streamlit as st
import pandas as pd
import numpy as np
from classifier import Classifier
from helpers import df_col_drop, preprocess_text, text_to_int, initialize_and_compute_matrices, \
    compute_and_normalize_counts
from sklearn.model_selection import train_test_split

# nltk.download('stopwords')


st.title('Two Poeters One Markov: The DNA of Poetry')


user_input = st.text_area("Enter your poem:", "Write your poem here")


df_cy = pd.read_json("data/poems_cy.json")
df_nh = pd.read_json("data/poems_nh.json")


df_cy = df_col_drop(df_cy, 'link')
df_nh = df_col_drop(df_nh, 'link')

df_cy['labels'] = 0
df_nh['labels'] = 1

df = pd.concat([df_cy, df_nh], ignore_index=True)
df['poem_text'] = df['poem_text'].apply(preprocess_text)


poems = df['poem_text']
labels = df['labels']

X_train, X_test, y_train, y_test = train_test_split(poems, labels, test_size=.2)


word2idx = {'<unk>': 0}
idx = 1
for text in X_train:
    tokens = text.split()
    for token in tokens:
        if token not in word2idx:
            word2idx[token] = idx
            idx += 1

X_train_int = [text_to_int(text, word2idx) for text in X_train]
X_test_int = [text_to_int(text, word2idx) for text in X_test]


A0, pi0, A1, pi1 = initialize_and_compute_matrices(X_train, y_train, word2idx)


logA0, logpi0 = compute_and_normalize_counts([t for t, y in zip(X_train, y_train) if y == 0], word2idx,token, A0, pi0)
logA1, logpi1 = compute_and_normalize_counts([t for t, y in zip(X_train, y_train) if y == 1], word2idx,token, A1, pi1)


clf = Classifier([logA0, logA1], [logpi0, logpi1],
                 np.log([sum(y_train == 0) / len(y_train), sum(y_train == 1) / len(y_train)]))

if st.button('Sınıflandır'):

    processed_input = preprocess_text(user_input)
    input_int = text_to_int(processed_input, word2idx)

    prediction = clf.predict([input_int])[0]

    if prediction == 0:
        st.write("This poem is probably by Can Yücel.")
    else:
        st.write("This poem is probably by Nazım Hikmet.")