#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 19:20:18 2018

@author: himanshu
"""

import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint




import sys

from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints
from sklearn.metrics import roc_auc_score


class Attention(Layer):
     # Input shape 3D tensor with shape: `(samples, steps, features)`.
     # Output shape 2D tensor with shape: `(samples, features)`.

    def __init__(self, step_dim,W_regulizer = None,b_regulizer = None,
                 W_constraint = None, b_constraint = None,bias = True,**kwargs):
        
        self.W_regulizer = W_regulizer
        self.b_regulizer = b_regulizer
        
        self.W_constraint = W_constraint
        self.b_constraint = b_constraint
        
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        self.init = initializers.get('glorot_uniform')
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1],),
                                      initializer= self.init,
                                      constraint = self.W_constraint,
                                      regularizer = self.W_regulizer,
                                      name = '{}_W'.format(self.name))
        
        self.features_dim = input_shape[-1]
        
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regulizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        super(Attention, self).build(input_shape)  

    
    def call(self, x, mask=None):
      
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
           
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
       




        
path = '../input/data'
path1 = '../input/glove-840b-tokens-300d-vectors/'
EMBEDDING_FILE=path1+'glove.840B.300d.txt'
TRAIN_DATA_FILE=path+'train.csv'
TEST_DATA_FILE=path+'test.csv'

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 300
num_dense = 256
lstm_dropout_rate = 0.25
dense_dropout_rate = 0.25

act = 'relu'


########################################
## index word vectors.
########################################
print('Indexing word vectors')
embedding_index = {}
with open(EMBEDDING_FILE,'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embedding_index[word] = coefs
print('Indexed the word vectors')   
print('Found %s word vectors.' %len(embedding_index))     

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)


########################################
## Basic preprocessing of text data. 
########################################
print('performing some basic preprocessing on data')

#regex for removing non-alphanumeric characters and spaces
remove_special_char = re.compile('r[^a-z\d]',re.IGNORECASE)

#regex to replace all numerics
replace_numerics = re.compile(r'\d+',re.IGNORECASE)


##############################################################################################
## fuction for coverting the text to list of tokens after stopword removal and stemming.
##############################################################################################
def preprocess_text(text, remove_stopwords = True, perform_stemming = True):
    #convert text to lowercase and split.
    text = text.lower().split()
    
    #stopword removal(you can use your own set of stopwords, here we are using default from nltk stopwords)
    if(remove_stopwords):
        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words]
     
    text = ' '.join(text)   
    
    text = remove_special_char.sub('', text)
    text = replace_numerics.sub('n', text)
        
    if(perform_stemming):
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = ' '.join(stemmed_words)
        
    return text    


##################################################
## forming sequeces to feed into the network.
##################################################    
raw_train_comments = train_df['comments'].fillna('NA').values
raw_test_comments = test_df['comments'].fillna('NA').values
classes_to_predict = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_df[classes_to_predict].values
#y_test_predicted = test_df[classes_to_predict].values

processed_train_comments = []
for comment in raw_train_comments:
    processed_train_comments.append(preprocess_text(comment))
    
processed_test_comments = []    
for comment in raw_test_comments:
    processed_test_comments.append(preprocess_text(comment))
        

tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(processed_train_comments + processed_test_comments)

train_sequences = tokenizer.text_to_sequences(processed_train_comments)
test_sequences = tokenizer.text_to_sequences(processed_test_comments)

print('found %s tokens in text.' %(tokenizer.word_index))

train_data = pad_sequences(train_sequences, maxlen = MAX_SEQUENCE_LENGTH)
final_test_data = pad_sequences(test_sequences, maxlen = MAX_SEQUENCE_LENGTH)

print('shape of train_data(will be divided further into final_train_data + final_validation_data) ready for feeding to network is %s' %(train_data.shape))
print('shape of final_test_data ready for fedding to network is %s' %(final_test_data.shape))
print('shape o