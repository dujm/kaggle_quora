#!/usr/bin/env python3
# %%
import os, math
import numpy as np
import pandas as pd
import itertools
import time
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import (
    Dense,
    Input,
    LSTM,
    Embedding,
    Dropout,
    Activation,
    GRU,
    Conv1D,
    Bidirectional,
)
from tensorflow.python.keras import (
    initializers,
    regularizers,
    constraints,
    optimizers,
    layers,
)

from sklearn.model_selection import train_test_split
from utils import standardize_text, text_to_array, batch_gen

# %% {"_uuid": "0e487c746161cdc8bfaf09a24dd1bb874afef115"}
pd.set_option('display.max_columns', 500)
tf.__version__


# %% [markdown] {"_uuid": "575ad3a0505f95215153b70f9dc84c2cd3c011fa"}
# ## 1.Explore dataframe features

# %%
def standardize_text(df, question_field):
    df[question_field] = df[question_field].str.replace(r"http\S+", "")
    df[question_field] = df[question_field].str.replace(r"http", "")
    df[question_field] = df[question_field].str.replace(r"@\S+", "")
    df[question_field] = df[question_field].str.replace(
        r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " "
    )
    df[question_field] = df[question_field].str.replace(r"@", "at")
    df[question_field] = df[question_field].str.lower()
    return df


# %%
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:30]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds += [empyt_emb] * (30 - len(embeds))
    return np.array(embeds)


# %% {"_uuid": "2aac670e30274b12dd31c8e06a5d403e45477baf"}
os.chdir('/Users/j/Dropbox/Learn/kaggle_quora/src/')
train_df = pd.read_csv("data/input/train.csv")
print(train_df.head())

test_df = pd.read_csv("data/input/test.csv")
print(test_df.head())


# 1. Preprocession: Lowercase, stemming, lemmarization, stopwords
train_clean = train_df.copy(deep=True)
test_clean = test_df.copy(deep=True)
train_clean = standardize_text(train_clean, 'question_text')
test_clean = standardize_text(test_clean, 'question_text')

# Split to train and validation data
train_df, val_df = train_test_split(train_clean, test_size=0.1)


# %%
# 2. Embdedding
# Source https://blog.tensorflow.python.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embeddings_index = {}
f = open('data/embeddings/glove.840B.300d/glove.840B.300d.txt')

for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# np.save('embeddings_index.npy', embeddings_index)
# embeddings_index = np.load('embeddings_index.npy', allow_pickle=True)


# 3. text_to_array
train_vects = [text_to_array(X_text) for X_text in tqdm(train_df["question_text"])]

val_vects = np.array(
    [text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])]
)

val_y = np.array(val_df["target"][:3000])


# %% {"_uuid": "c06041b7ee5792058819c138cc63b825cfa189d3"}
checkpoint_path = "./model/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, save_weights_only=True, verbose=1
)

# %% {"_uuid": "ef50d94d338ef4e7e2309ce12c49e0fb3b811578"}
mg = batch_gen(train_df)


def create_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(30, 300)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()
model.fit_generator(
    mg,
    epochs=5,
    steps_per_epoch=1000,
    validation_data=(val_vects, val_y),
    calllbacks=[cp_callback],
)
model.summary()

loss, acc = model.evaluate(val_vects, val_y)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
# Untrained model, accuracy: 96.43%

# %% TF 2.0 saved_model
# https://www.tensorflow.org/alpha/guide/saved_model
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/saved_model/save
tf.saved_model.save(model, "sincere/1/")
# saved_model_cli show --dir sincere/1 --tag_set serve --signature_def serving_default

# %%
# If you want to save the weights manually
# https://www.tensorflow.org/tutorials/keras/save_and_restore_models
# model.save("../models/tf-df/model{}.h5".format(int(time.time())))
# model.save_weights("../models/tf-df/weights{}".format(int(time.time())))


# Restore the weights
# model = create_model()
# model.load_weights('models/tf-df/weights1557089352.index')
