#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
# kaggle competitions download -c quora-insincere-questions-classification

# kaggle competitions files quora-insincere-questions-classification

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
from tensorflow.python.keras import export_saved_model
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
pd.set_option('display.max_columns', 500)
tf.__version__

# %% [markdown] {"_uuid": "575ad3a0505f95215153b70f9dc84c2cd3c011fa"}
# ## 1.Explore dataframe features

# %% {"_uuid": "2aac670e30274b12dd31c8e06a5d403e45477baf"}

os.listdir()
train_df = pd.read_csv("input/train.csv")
print(train_df.head())
# 1. fill up the missing values
test_df =pd.read_csv("input/test.csv")

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
# Source https://blog.tensorflow.python.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embeddings_index = {}
f = open('embeddings/glove.840B.300d/glove.840B.300d.txt')
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
              metrics=['accuracy'])
    return model

model =create_model()
model.summary()
# %% {"_uuid": "f344ba26a37fa4ac83f7f3104bee41bff5617f82"}
mg = batch_gen(train_df)
model.fit_generator(mg, epochs=5,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),
                    verbose=True)
# ensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# %% {"_uuid": "d1c01b5a704854c36229996b3f894e5f9bfb9092"}
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
# Save the weights manually 
# https://www.tensorflow.org/tutorials/keras/save_and_restore_models
model.save("../models/tf-df/model{}.h5".format(int(time.time())))

model.save_weights("../models/tf-df/weights{}".format(int(time.time())))

# Restore the weights
model = create_model()
model.load_weights('../models/tf-df/weights1557087325.index')

loss, acc = model.evaluate(val_vects, val_y)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
# Untrained model, accuracy: 96.43%


# %% {"_uuid": "b8a016529af0d8ff2bcd7887ba20f5de110f81c1"}
# Tensorflow serving 
# A SavedModel contains a complete TensorFlow program, including weights and computation. 
# cd /Users/j/Dropbox/Learn/tensorflow2.0/serving

# TF 2.0 saved_model 
# https://www.tensorflow.org/alpha/guide/saved_model
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/saved_model/save
'''
# to delete 
tf.saved_model.save(model, "../models/tf-df/tmp/1/")

!saved_model_cli show --dir ../models/tf-df/tmp/1 --tag_set serve --signature_def serving_default

The given SavedModel SignatureDef contains the following input(s):
  inputs['bidirectional_2_input'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 30, 300)
      name: serving_default_bidirectional_2_input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['dense_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict

'''
# use a name to test mobilenet 
!saved_model_cli show --dir /tmp/mobilenet/1 --tag_set serve --signature_def serving_default
  
# Working

docker run -p 8500:8500 \
--mount type=bind,source=/tmp/mobilenet,target=/models/mobilenet \
-e MODEL_NAME=mobilenet -t tensorflow/serving &

  
# Test data
test_samll= test_df[0:3]
small = test_samll['question_text'].values.tolist()

import json
data = json.dumps({"signature_name": "serving_default", "instances": small })

print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))


##############################################################################
!pip install -q requests

import requests
headers = {"content-type": "application/json"}

!curl -d '{"instances": small}' \
    -X POST http://localhost:8500/v1/models/mobilenet:predict
    --output


curl -d '{"instances": ["Why do so many women become so rude and arrogant when they get just a little bit of wealth and power?","When should I apply for RV college of engineering and BMS college of engineering?]}' \
    -X POST http://localhost:8500/v1/models/mobilenet:predict \
    -o output2.txt 
    
# working but empty     
json_response = requests.post('http://localhost:8501/v1/models/mobilenet:predict', data=data, headers=headers)


predictions = json.loads(json_response.text)['predictions']

show(0, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
  class_names[np.argmax(predictions[0])], test_labels[0], class_names[np.argmax(predictions[0])], test_labels[0]))
##############################################################################
# Not working 
'''
loaded = tf.saved_model.load("1/")
print(list(loaded.signatures.keys()))  # ["serving_default"]
'''
##############################################################################



##############################################################################
#prediction on test file for kaggle submission 

batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr


all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(model.predict(x).flatten())

# %% {"_uuid": "ba5fd8920241e54810f5d6bdbfd8b6f4bbf71da2"}
y_te = (np.array(all_preds) > 0.5).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("./output/submission.csv", index=False)

!head data/output/submission.csv

!head data/input/test.csv   
'''
!head data/input/test.csv
qid,question_text
0000163e3ea7c7a74cd7,Why do so many women become so rude and arrogant when they get just a little bit of wealth and power?
00002bd4fb5d505b9161,When should I apply for RV college of engineering and BMS college of engineering? Should I wait for the COMEDK result or am I supposed to apply before the result?
00007756b4a147d2b0b3,What is it really like to be a nurse practitioner?
000086e4b7e1c7146103,Who are entrepreneurs?
0000c4c3fbe8785a3090,Is education really making good people nowadays?
000101884c19f3515c1a,How do you train a pigeon to send messages?
00010f62537781f44a47,What is the currency in Langkawi?
00012afbd27452239059,"What is the future for Pandora, can the business reduce its debt?"
00014894849d00ba98a9,My voice range is A2-C5. My chest voice goes up to F4. Included sample in my higher chest range. What is my voice type?
'''