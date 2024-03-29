import numpy as np
import re
import unidecode
from TurkishStemmer import TurkishStemmer
from nltk.corpus import stopwords

def df_col_drop(df,x):
    df = df.drop(x,axis=1)
    return df
def tidy_characters(text):
    return unidecode.unidecode(text)
def lemmatize_words(text):
    stemmer = TurkishStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])
def preprocess_text(text):
    text = tidy_characters(text)
    text = re.sub('[^a-z A-z 0-9-]+', '', text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('turkish')])
    text = text.rstrip().lower()
    text = lemmatize_words(text)
    return text
def text_to_int(text, word2idx):
    tokens = text.split()
    return [word2idx.get(token, 0) for token in tokens]  # <unk> için 0 kullanılır
def initialize_and_compute_matrices(X_train, y_train, word2idx):
    V = len(word2idx)  # Sözlük büyüklüğü

    # Laplace düzeltmesi ile matrislerin başlatılması
    A0 = np.ones((V, V))  # Sınıf 0 için geçiş matrisi
    pi0 = np.ones(V)  # Sınıf 0 için başlangıç olasılık vektörü

    A1 = np.ones((V, V))  # Sınıf 1 için geçiş matrisi
    pi1 = np.ones(V)  # Sınıf 1 için başlangıç olasılık vektörü

    # Matrisleri doldurma
    for text, label in zip(X_train, y_train):
        tokens = text.split()
        indices = [word2idx.get(token, 0) for token in tokens]  # Sözlükte olmayan kelimeler için <unk>

        for i in range(len(indices) - 1):
            if label == 0:
                A0[indices[i], indices[i + 1]] += 1
                pi0[indices[0]] += 1  # İlk kelimenin olasılığını artır
            else:
                A1[indices[i], indices[i + 1]] += 1
                pi1[indices[0]] += 1  # İlk kelimenin olasılığını artır

    # Olasılıkları normalize etme
    A0 /= A0.sum(axis=1, keepdims=True)
    pi0 /= pi0.sum()

    A1 /= A1.sum(axis=1, keepdims=True)
    pi1 /= pi1.sum()

    return A0, pi0, A1, pi1

def compute_and_normalize_counts(text_as_int,word2idx,token, A, pi):
    for tokens in text_as_int:
        last_idx = None
        for idx in tokens:
            idx = word2idx.get(token, 0)
            if last_idx is None:
                pi[idx] += 1
            else:
                A[last_idx, idx] += 1
            last_idx = idx
    A /= A.sum(axis=1, keepdims=True)
    pi /= pi.sum()
    return np.log(A), np.log(pi)