# This Python 3 environment comes with many helpful analytics libraries installed
# Any results you write to the current directory are saved as output.
## Install pytorch-pretrained-bert
#!pip install --upgrade pip
#!pip install keras-bert
#!pip install pytorch-pretrained-bert

##Ref: https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
# Load the data
#####
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
#####

train_df = pd.read_csv("../input/train.csv").sample(frac=1, random_state=40).reset_index(drop=True)

test_df = pd.read_csv("../input/test.csv", index_col='qid')

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


# 4. Tokenizing sentences to a list of separate words that the model can understand
tokenizer = RegexpTokenizer(r'\w+') #\w matches 0..+inf word characters

train_clean["tokens"] =train_clean['question_text'].apply(tokenizer.tokenize)
print(train_clean.head())

test_clean["tokens"] =test_clean['question_text'].apply(tokenizer.tokenize)
print(test_clean.head())




#6. Prepare the sentences and labels
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam


## 6.1 fix some configurations.
#### Limit our sequence length to 75 tokens
#### use a batch size of 32 as suggested by the Bert paper.
##### Note, that Bert natively supports sequences of up to 512 tokens.

MAX_LEN = 75
bs = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
'Tesla K80'

"""The Bert implementation comes with a pretrained tokenizer
and a definied vocabulary.
We load the one related to the smallest pre-trained model bert-base-uncased.
Try also the cased variate since it is well suited for NER.

"""
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# 5 Parepare index
targets_vals = list(set(train_clean["target"].values))
target2idx = {t: i for i, t in enumerate(targets_vals)}
target2idx

## 6.  make train corpus
train_list_corpus = train_clean["question_text"].tolist()
train_list_labels = train_clean["target"].values.tolist()
train_list_labels2 = train_clean["target"].values.tolist()
targets_vals


# prepare test samples
test_list_corpus= test_clean["question_text"].tolist()

## 6.3  tokenize all sentences
tokenized_texts = [tokenizer.tokenize(sent) for sent in train_list_corpus]
print(tokenized_texts[0])
"""
['how', 'come', 'canada', 'is', 'able', 'to', 'avoid', 'the', 'wave', 'of', 'alt', 'right', 'pop', '##uli', '##sm', 'sweeping', 'western', 'demo', '##cr', '##acies', '?']
"""

## 6.4 Cut and pad the token and label sequences to our desired length.
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

"""Not working"""
targets = pad_sequences([[target2idx.get(l) for l in lab] for lab in [train_list_labels2]],
                     maxlen=MAX_LEN, value=target2idx["O"], padding="post",
                     dtype="long", truncating="post")

"""'int' object is not iterable"""


## The Bert model supports something called attention_mask,
### which is similar to the masking in keras.
### So here we create the mask to ignore the padded elements in the sequences

attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
targets = train_list_labels
## 6.5
tr_inputs, val_inputs, tr_targets, val_targets = train_test_split(input_ids, targets,
                                                            random_state=40, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=40, test_size=0.1)





tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_targets = torch.tensor(tr_targets)
val_targets = torch.tensor(val_targets)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)





train_data = TensorDataset(tr_inputs, tr_masks, tr_targets)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_targets)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)




model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(target2idx))

model.cuda();



FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)




from sklearn.metrics import roc_curve, precision_recall_curve, f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(target2idx))


epochs = 5
max_grad_norm = 1.0






""" Expected input batch_size (2400) to match target batch_size (32)."""

print(tr_inputs.shape)
print(val_inputs.shape)
print(tr_targets.shape)
print(val_targets.shape)
print(tr_masks.shape)
print(val_masks.shape)

"""torch.Size([1175509, 75])
torch.Size([130613, 75])
torch.Size([1175509])
torch.Size([130613])
torch.Size([1175509, 75])
torch.Size([130613, 75])"""

"""Shut down GPU" also does not solve the problem
