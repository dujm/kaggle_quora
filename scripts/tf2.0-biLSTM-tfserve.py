#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:24:43 2019

@author: j
"""

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

# Install nltk 
#!conda install -c anaconda nltk 
# Download data 
#cd path/src/data/input
# kaggle competitions download -c quora-insincere-questions-classification
# Convert 
import os, math
import numpy as np 
import pandas as pd 
import itertools
import time

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D,Bidirectional
from tensorflow.python.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import initializers, regularizers, constraints, optimizers, layers

#nltk
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

from tqdm import tqdm
from sklearn.model_selection import train_test_split
##############################################################################
# %% {"_uuid": "b67995f2cd04efbd86747fbca05dea1450f82e8b"}

# %% {"_uuid": "0e487c746161cdc8bfaf09a24dd1bb874afef115"}
pd.set_option('display.max_columns', 500)
tf.__version__

# %% [markdown] {"_uuid": "575ad3a0505f95215153b70f9dc84c2cd3c011fa"}
# ## 1.Explore dataframe features

# %% {"_uuid": "2aac670e30274b12dd31c8e06a5d403e45477baf"}
os.chdir('/Users/j/Dropbox/Learn/kaggle_quora/src/')
os.listdir()
train_df = pd.read_csv("data/input/train.csv")
print(train_df.head())

test_df =pd.read_csv("data/input/test.csv")
print(test_df.head())

# 2. Are there overlaps between train and test? No
print(pd.core.common.intersection(train_df['question_text'], test_df['question_text']).tolist())
print(pd.core.common.intersection(train_df['qid'], test_df['qid']).tolist())
# Split to train and validation data
train_df, val_df = train_test_split(train_clean, test_size=0.1)

##############################################################################
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

train_clean = train_df.copy(deep=True) # modification of the orginial df will not be affected 
test_clean = test_df.copy(deep=True)
train_clean = standardize_text(train_clean, 'question_text')
test_clean = standardize_text(test_clean, 'question_text')

# %% {"_uuid": "9a70477a85aca0d9df05ac6a6bbc80467c374998"}
# Embdedding setup
# Source https://blog.tensorflow.python.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

##############################################################################
embeddings_index = {}
f = open('data/embeddings/glove.840B.300d/glove.840B.300d.txt')

for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
np.save('embeddings_index.npy', embeddings_index)
embeddings_index = np.load('embeddings_index.npy', allow_pickle=True)


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


##############################################################################
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


##############################################################################
# %% {"_uuid": "c06041b7ee5792058819c138cc63b825cfa189d3"}
checkpoint_path = "./model/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# %% {"_uuid": "ef50d94d338ef4e7e2309ce12c49e0fb3b811578"}
def create_model():            
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True),
                        input_shape=(30, 300)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )
    return model

model =create_model()

# %% {"_uuid": "f344ba26a37fa4ac83f7f3104bee41bff5617f82"}
mg = batch_gen(train_df)
model.fit_generator(mg, epochs=5,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),
                    calllbacks =[cp_callback])
model.summary()
'''
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_2 (Bidirection (None, 30, 128)           186880    
_________________________________________________________________
bidirectional_3 (Bidirection (None, 128)               98816     
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 129       
=================================================================
Total params: 285,825
Trainable params: 285,825
Non-trainable params: 0
________________________________
'''

##############################################################################
# If you want to save the weights manually 
# https://www.tensorflow.org/tutorials/keras/save_and_restore_models
model.save("../models/tf-df/model{}.h5".format(int(time.time())))

model.save_weights("../models/tf-df/weights{}".format(int(time.time())))

##############################################################################
# Restore the weights
model = create_model()
model.load_weights('models/tf-df/weights1557089352.index')

loss, acc = model.evaluate(val_vects, val_y)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
# Untrained model, accuracy: 96.43%
##############################################################################
# %% {"_uuid": "b8a016529af0d8ff2bcd7887ba20f5de110f81c1"}
# Tensorflow serving 
# A SavedModel contains a complete TensorFlow program, including weights and computation. 

# TF 2.0 saved_model 
# https://www.tensorflow.org/alpha/guide/saved_model
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/saved_model/save
!saved_model_cli show --dir ../models/models/sincere/1 --tag_set serve --signature_def serving_default

##############################################################################
# Build tensorflow serving 
git clone https://github.com/tensorflow/serving.git
cd serving
tools/run_in_docker.sh bazel build -c opt tensorflow_serving/...    
# Binaries are placed in the bazel-bin directory, and can be run using a command like
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server

# To test your build, execute
tools/run_in_docker.sh bazel test -c opt tensorflow_serving/...
##############################################################

# tf serving using docker
# Serve /Users/j/Dropbox/Learn/kaggle_quora/src/models/sincere
docker run -t --rm -p 8501:8501 \
    -v "/Users/j/Dropbox/Learn/kaggle_quora/src/models/sincere:/models/sincere" \
    -e MODEL_NAME=sincere \
    tensorflow/serving &

# Check http://localhost:8501/v1/models/sincere, confirms working 
'''
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}

'''
##############################################################################

