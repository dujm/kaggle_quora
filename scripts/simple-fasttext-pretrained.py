# -*- coding: utf-8 -*-
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

# %% [markdown] {"_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19", "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5"}
# ### Reference 
# https://www.kaggle.com/mabrek/simple-fasttext-pretrained

# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
import warnings
import traceback
import sys
from datetime import datetime
import json

import numpy as np
import pandas as pd
from timeit import default_timer as timer

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint as randint
from scipy.stats import uniform as uniform
from sklearn.utils import check_random_state

import fastText

# %% {"_uuid": "aa3601c93dd14e69676e998d5ec647155b117bb2"}
PUNCTS_FASTTEXT = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '{', '}', '©', '^', '®',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]


def clean_fasttext(x):
    x = str(x)
    for punct in PUNCTS_FASTTEXT:
        x = x.replace(punct, f' {punct} ')
    return x



# %% {"_uuid": "0f2d652e6e7393800da3bcc36fa29ea740f8315c"}
def predict_fasttext_single(model, x):
    labels, probs = model.predict(x, 2)
    if labels[0] == '__label__1':
        return probs[0]
    else:
        return probs[1]


def predict_fasttext(model, df):
    return df.cleaned_text.apply(lambda x: predict_fasttext_single(model, x))


# from https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/76391
def scoring(y_true, y_proba, verbose=True):
    from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    def threshold_search(y_true, y_proba):
        precision , recall, thresholds = precision_recall_curve(y_true, y_proba)
        thresholds = np.append(thresholds, 1.001) 
        F = 2 / (1/precision + 1/recall)
        best_score = np.max(F)
        best_th = thresholds[np.argmax(F)]
        return best_th 


    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

    scores = []
    ths = []
    for train_index, test_index in rkf.split(y_true, y_true):
        y_prob_train, y_prob_test = y_proba[train_index], y_proba[test_index]
        y_true_train, y_true_test = y_true[train_index], y_true[test_index]

        # determine best threshold on 'train' part 
        best_threshold = threshold_search(y_true_train, y_prob_train)

        # use this threshold on 'test' part for score 
        sc = f1_score(y_true_test, (y_prob_test >= best_threshold).astype(int))
        scores.append(sc)
        ths.append(best_threshold)

    best_th = np.mean(ths)
    score = np.mean(scores)

    if verbose: print(f'Best threshold: {np.round(best_th, 4)}, Score: {np.round(score,5)}')

    return best_th, score

# %% {"_uuid": "f268f9c96514521ecb0a03cc848707cc07dd9696"}
# !mkdir -p ../tmp

# %% {"_uuid": "8bc80f867fd758cd7e9ca464ee63a20ba9e718f4"}
train = pd.read_csv("../input/train.csv").sample(frac=1, random_state=3465).reset_index(drop=True)

train['cleaned_text'] = train["question_text"].apply(lambda x: clean_fasttext(x)).str.replace('\n', ' ')
fasttext_labeled = '__label__' + train.target.astype(str) + ' ' + train.cleaned_text

np.savetxt('../tmp/train.txt', fasttext_labeled.values, fmt='%s')


test = pd.read_csv("../input/test.csv", index_col='qid')
test['cleaned_text'] = test["question_text"].apply(lambda x: clean_fasttext(x)).str.replace('\n', ' ')

# %% {"_uuid": "89b9518252e6df159632da0bf14b0aab8d5988aa"}
parameters = {
        'lr': 0.161195,
        'dim': 300,
        'ws': 5,
        'epoch': 10,
        'minCount': 80,
        'minCountLabel': 0,
        'minn': 4,
        'maxn': 5,
        'neg': 5,
        'wordNgrams': 3,
        'loss': "hs",
        'bucket': 2000000,
        'thread': 4,
        'lrUpdateRate': 100,
        't': 1e-4,
        'pretrainedVectors': '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
        'verbose': 0
    }

# %% {"_uuid": "619af4cc00c03793f1462a17828933c3f3889b3b"}
model = fastText.train_supervised(input='../tmp/train.txt', **parameters)

# %% {"_uuid": "7b6e4617d29582d312bcea09a0af4778203a1014"}
train_pred = predict_fasttext(model, train)
test_pred = predict_fasttext(model, test)

# %% {"_uuid": "bf732b49796f95ff08021d42100352c51618133b"}
best_th, f1 = scoring(train.target, train_pred, verbose=False)

# %% {"_uuid": "bdb6e9f92f3d2991b63ed550d493e4f1e0d04890"}
best_th, f1

# %% {"_uuid": "06a582fe376be94b22f363a2a114fa7d7ad60a6a"}
pred = (test_pred > best_th).astype('int').rename('prediction')

# %% {"_uuid": "98f7823e657b30a1522805886b390abab77e9aa6"}
pd.DataFrame(pred).to_csv('submission.csv')

# %% {"_uuid": "e46bb64de82fc675e25b1bce8d14e7f32a4ce482"}

# %% {"_uuid": "9e3ba5c942b7c78f16d8caf7aa48b85bb2dc7090"}
