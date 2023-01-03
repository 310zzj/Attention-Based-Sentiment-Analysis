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
EMBEDDING_