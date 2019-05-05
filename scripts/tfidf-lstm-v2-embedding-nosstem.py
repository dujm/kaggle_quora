# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% {"_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19", "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5"}
import os, math
import numpy as np 
import pandas as pd 
import itertools


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM, Dense
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
#nltk model 
from nltk.tokenize import RegexpTokenizer

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# %% {"_uuid": "b67995f2cd04efbd86747fbca05dea1450f82e8b"}

# %% {"_uuid": "0e487c746161cdc8bfaf09a24dd1bb874afef115"}
#pd.set_option('display.height', 1000)
#pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)

# %% [markdown] {"_uuid": "575ad3a0505f95215153b70f9dc84c2cd3c011fa"}
# ## 1.Explore dataframe features

# %% {"_uuid": "2aac670e30274b12dd31c8e06a5d403e45477baf"}
train_df = pd.read_csv("../input/train.csv")
# 1. fill up the missing values
test_df =pd.read_csv("../input/test.csv")
print(train_df.head())
print(test_df.head())

# 2. Are there overlaps between train and test? No
print(pd.core.common.intersection(train_df['question_text'], test_df['question_text']).tolist())
print(pd.core.common.intersection(train_df['qid'], test_df['qid']).tolist())

#3 Some data features
# print('train data',train_df.info())
#print('test data',test_df.info())
#Are there replicated rows? No
#print(train_df.nunique())



# %% [markdown] {"_uuid": "eb524f082e35bb9a43c53a07950f8fd45639e67d"}
# ## 2. Feature extraction from text

# %% {"_uuid": "f13631ca516d65b2d60c8aeee26b93464cfda542"}
#1. Preprocession: Lowercase, stemming, lemmarization, stopwords
def standardize_text(df, question_field):
    df[question_field] = df[question_field].str.replace(r"http\S+", "")
    df[question_field] = df[question_field].str.replace(r"http", "")
    df[question_field] = df[question_field].str.replace(r"@\S+", "")
    df[question_field] = df[question_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[question_field] = df[question_field].str.replace(r"@", "at")
    df[question_field] = df[question_field].str.lower()
    return df
# 2. 
train_clean = train_df.copy(deep=True) # modification of the orginial df will not be affected 
test_clean = test_df.copy(deep=True)
train_clean = standardize_text(train_clean, 'question_text')
test_clean = standardize_text(test_clean, 'question_text')
# 3. Are there overlaps between train and test question_text after preprocession? Yes
print(pd.core.common.intersection(train_clean['question_text'], test_clean['question_text']).tolist())


# %% {"_uuid": "9a70477a85aca0d9df05ac6a6bbc80467c374998"}
# embdedding setup
# Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# %% {"_uuid": "8f4912b409bd07c66b9b32f99212ad771ba61a63"}
train_df, val_df = train_test_split(train_clean, test_size=0.1)


# %% {"_uuid": "dddcc226926e60d9eb674ef1c92734a4fa3058ab"}
# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:30]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (30 - len(embeds))
    return np.array(embeds)

# train_vects = [text_to_array(X_text) for X_text in tqdm(train_df["question_text"])]
val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])
val_y = np.array(val_df["target"][:3000])

# %% {"_uuid": "4f2311e86d07936b88f8820928cc9adca4633a1f"}
# Data providers
batch_size = 64

def batch_gen(train_df):
    n_batches = math.ceil(len(train_df) / batch_size)
    while True: 
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])


# %% {"_uuid": "c06041b7ee5792058819c138cc63b825cfa189d3"}
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional

# %% {"_uuid": "ef50d94d338ef4e7e2309ce12c49e0fb3b811578"}
model = Sequential()
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True),
                        input_shape=(30, 300)))
model.add(Bidirectional(CuDNNLSTM(64)))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# %% {"_uuid": "f344ba26a37fa4ac83f7f3104bee41bff5617f82"}
mg = batch_gen(train_df)
model.fit_generator(mg, epochs=20,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),
                    verbose=True)

# %% {"_uuid": "d1c01b5a704854c36229996b3f894e5f9bfb9092"}
#prediction part
batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

test_df = pd.read_csv("../input/test.csv")

all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(model.predict(x).flatten())

# %% {"_uuid": "ba5fd8920241e54810f5d6bdbfd8b6f4bbf71da2"}
y_te = (np.array(all_preds) > 0.5).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)

# %% {"_uuid": "cad4d9a7345aff3603e6a65f55b75625fd187d51"}
# !head submission.csv

# %% {"_uuid": "b8a016529af0d8ff2bcd7887ba20f5de110f81c1"}
