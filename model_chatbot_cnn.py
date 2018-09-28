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

import os
#from tflearn.data_utils import to_categorical, pad_sequences

import pandas as pd
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
with open("C:/Users/Magnifier/Desktop/data/machine_learning/cnn_chatbot/objects.pickle", 'rb') as f:
    og = pickle.load(f)
    
with open("C:/Users/Magnifier/Desktop/data/machine_learning/cnn_chatbot/word_dict.pickle", 'rb') as f:
    word_dict = pickle.load(f)

with open("C:/Users/Magnifier/Desktop/data/machine_learning/cnn_chatbot/words.pickle", 'rb') as f:
	words = pickle.load(f)
    
with open("C:/Users/Magnifier/Desktop/data/machine_learning/cnn_chatbot/categories.pickle", 'rb') as f:
	categories = pickle.load(f)

#dfset = pad_sequences(bracket, maxlen=np.max(checklen), value=0.)



# method to remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)
 
#initialize the stemmer
stemmer = LancasterStemmer()
### Step 3: Convert the data into Tensorflow specification

## create our training data
#all_data = []
#output = []
## create an empty array for our output
#output_empty = [0] * len(categories)
# 
# 
#for doc in docs:
#    # initialize our bag of words(bow) for each document in the list
#    bow = []
#    # list of tokenized words for the pattern
#    token_words = doc[0]
#    # stem each word
#    token_words = [stemmer.stem(word.lower()) for word in token_words]
#    # create our bag of words array
#    for w in words:
#        bow.append(1) if w in token_words else bow.append(0)
#     
#    output_row = list(output_empty)
#    output_row[categories.index(doc[1])] = 1
#     
#    # our training set will contain a the bag of words model and the output row that tells which catefory that bow belongs to.
#    all_data.append([bow, output_row])
 
# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(og)
all_data = np.array(og)


test_size = 0.1
testing_size = int(test_size * len(all_data)) 
# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(all_data[:,0][:-testing_size])
train_y = list(all_data[:,1][:-testing_size])


#train_x = list(all_data[:,0])
#train_y = list(all_data[:,1])

#train_x = list(all_data[:,0][:-testing_size])
##train_y = all_data[:, [-1]][:-testing_size]
test_x = list(all_data[:,0][-testing_size:])
test_y = list(all_data[:,1][-testing_size:])


###Step 4: Initiate Tensorflow text classification

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
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
#model.fit(train_x, train_y, n_epoch = 20, shuffle=True, show_metric=True, batch_size=32)

model.fit(train_x, train_y, n_epoch = 20, shuffle=True, validation_set=(test_x, test_y), show_metric=True, batch_size=100)

# conv = [word_dict[w] for w in token_words]
def get_tf_record(sentence):
    global words
# tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
# stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
# bag of words
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

#x = 1
#while x >= 1:
#    try:
#        print('Hey, Do you want me to answer your multiple V.Ns?')
#        print('Actually It is a piece of cake haha. Just attach your csv file!')
#        dest_dir = os.path.expandvars('%UserProfile%/Desktop/data/machine_learning/cnn_chatbot')
#        quest = input('Write down your csv file name(abc.csv):')
#        test = pd.read_csv(dest_dir+'/'+quest, sep=",", encoding='ISO-8859-1')
#        answ = []
#        for i in test['VulnerabilityName']:
#            hc = model.predict([get_tf_record(i)])
#            cd = np.argmax(hc)
#            score = hc[0][cd]
#            if np.max(hc) > 0.4:
#                cat = categories[np.argmax(hc)]
#                answ.append(cat)
#            else:
#                answ.append('not configured')
#            print([i,cat,score])    
#        
#        test['category'] = answ
#        print('Do you want to export my answers into CSV file')
#        quest1 = input('please tell me yes or no(Y or N):')
#        if quest1 == "yes" or quest1 =="Yes" or quest1 =="y" or quest1 =="Y":
#            filename = os.path.join(os.environ["HOMEDRIVE"], os.environ["HOMEPATH"], "Desktop/aichatbot", 'myanswersheet.csv')
#
#            f = open(filename,'a')        
#            test.to_csv(f,index=False)
#            f.close()
#            print("Exported. You can check your folder:Desktop/aichatbot")     
#        else:
#            print('Haha I see, next time!')
#        print('.')
#        print('.')
#    except:
#        print('your file name(csv) might be wrong. Please try It again')
#        print('*')
#        print('*')
#    x += 1
#print('***There are 7 vulnerability categories as below:***')
#print ("\n".join([str(x) for x in categories]))
#
#
#sent_1 = 'sangwoo sangwoo CIFS NULL'
##sent-2 sensitive data exposure
#sent_2 = 'FTP access with anonymous account' 
##sent_2 = 'FTP access sangwoo with anonymous account sangwoo' 
#
#sent-3 'using components with known vulnerabilities'


#sent_3 = 'FTP credentials transmitted unencrypted'
#percent = model.predict([get_tf_record(sent_3)])
#print(percent)
#print( categories[np.argmax(model.predict([get_tf_record(sent_3)]))])
##'Back', 'Orifice', 'Backdoor', 'Installed'
#
#dest_dir = os.path.expandvars('%UserProfile%/Desktop/data/machine_learning/cnn_chatbot')
#quest = input('Write down your csv file name(abc.csv):')
#test = pd.read_csv(dest_dir+'/'+quest, sep=",", encoding='ISO-8859-1')
#answ = []
#for i in test['VulnerabilityName']:
#    hc = model.predict([get_tf_record(i)])
#    cd = np.argmax(hc)
#    score = hc[0][cd]
#    if np.max(hc) > 0.4:
#        cat = categories[np.argmax(hc)]
#        answ.append(cat)
#    else:
#        answ.append('not configured')
#    print([i,cat,score])    

