# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:01:41 2018

@author: SANGWOOLEE2
"""

print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('└[∵┌]└[ ∵ ]┘[┐∵]┘')
print('Hi I am an Artificial intelligence chatbot for cyber security.')
print('I am being trained to answer your questions. It takes about 2 ~ 3 mins.')
print('I will let you know when we are ready to talk to each other.')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression


import numpy as np
import pickle
import sys
import unicodedata
#import json
import nltk

from nltk.stem.lancaster import LancasterStemmer
import random

tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))


#Load the dataset
with open("path/objects.pickle", 'rb') as f:
    og = pickle.load(f)
    
with open("path/word_dict.pickle", 'rb') as f:
    word_dict = pickle.load(f)

with open("path/words.pickle", 'rb') as f:
	words = pickle.load(f)
    
with open("path/categories.pickle", 'rb') as f:
	categories = pickle.load(f)




# Remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)
 
#initialize the stemmer
stemmer = LancasterStemmer()



 
# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(og)
all_data = np.array(og)


test_size = 0.1
testing_size = int(test_size * len(all_data)) 
train_x = list(all_data[:,0][:-testing_size])
train_y = list(all_data[:,1][:-testing_size])


test_x = list(all_data[:,0][-testing_size:])
test_y = list(all_data[:,1][-testing_size:])


###Step 4: Feed the dataset into tflearn text classification model

# reset underlying graph data
tf.reset_default_graph()
# Build CNN
network = input_data(shape=[None, len(train_x[0])], name='input')
network = tflearn.embedding(network, input_dim=len(words), output_dim=100)
branch1 = conv_1d(network, 100, 3, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, 100, 4, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, 100, 5, padding='valid', activation='relu', regularizer="L2")
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, len(train_y[0]), activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)


model.fit(train_x, train_y, n_epoch = 20, shuffle=True, validation_set=(test_x, test_y), show_metric=True, batch_size=100)


def get_tf_record(sentence):
    global words

    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    bow = []
    for s in sentence_words:
        if s in words:
            bow.append(word_dict[s])
        else:
            bow.append(0)
    bows = np.pad(bow, pad_width=(0, 50-len(sentence_words)), mode='constant')
    return(np.array(bows))




print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('└[∵┌]└[ ∵ ]┘[┐∵]┘')
print('Hi I am an Artificial intelligence chatbot for cyber security.')
print('Thank you for your patience, how can I help you?')
print('If you write down vulnerability name(V.N), I will match your V.N with one of categories.')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('There are 7 vulnerability categories as below:')
print ("\n".join([str(x) for x in categories]))
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#print('Key in your input as below')
print('*')
print('*')
print('*')


