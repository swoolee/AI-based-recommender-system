# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:01:41 2018

@author: SANGWOOLEE2
"""



#from tflearn.data_utils import to_categorical, pad_sequences

#import pandas as pd
import numpy as np
import sys
import unicodedata
#import json
import nltk
import pickle

from nltk.stem.lancaster import LancasterStemmer


from pymongo import MongoClient

### Step 2: Data Loadand Pre-processing
# a table structure to hold the different punctuation used
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))



client = MongoClient('localhost', 00000)

db = client.folder



# Extract the values from mongo db query
df = db.abcd.aggregate([
      {'$project':
          {
            'Category': { '$toLower': '$Category' },
            'Name': { '$toLower': '$Name' }
          }}, 
      {'$group':
            {'_id': 
                {'Category': '$Category', "Name":'$Name'},
                 'uniqueCount': {'$addToSet': "$Name"}}},
      {'$project':
            {#"Vulnerability_Category":1,
             'uniqueCustomerCount':{'$size':"$uniqueCount"}}} 
]); 


# method to remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)
 
#initialize the stemmer
stemmer = LancasterStemmer()

data = None
 
def makeDic(words):
    Words = [stemmer.stem(w.lower()) for w in words]
    Words.insert(0,'')
    Words = list(set(Words))
    return Words



# a list of tuples with words in the name and category  
words = []
docs = []
categories = [] 


for y in df:
    each_sentence = remove_punctuation(y['Name'])
    w = nltk.word_tokenize(each_sentence)
    words.extend(w)
    docs.append((w, y['Category']))
    categories.append(y['Category'])


category = list(np.unique(categories))



word_dict = {w: i for i, w in enumerate(makeDic(words))}
#print (words)



output_empty = [0] * len(category)

#Convert words into number with padding
bracket = []
for doc,y in docs:
    token_words = doc
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    conv = [word_dict[w] for w in token_words]
    convs = np.pad(conv, pad_width=(0, 50-len(conv)), mode='constant')
    output_row = list(output_empty)
    output_row[category.index(y)] = 1
    bracket.append([convs,output_row])



#Store components(words, bracket, word_dict, category) into pickle files
with open("C:/Users/Magnifier/Desktop/data/machine_learning/cnn_chatbot/words.pickle", 'wb') as f:
	pickle.dump(words, f)

with open("C:/Users/Magnifier/Desktop/data/machine_learning/cnn_chatbot/objects.pickle", 'wb') as f:
	pickle.dump(bracket, f)

with open("C:/Users/Magnifier/Desktop/data/machine_learning/cnn_chatbot/word_dict.pickle", 'wb') as f:
	pickle.dump(word_dict, f)
    
with open("C:/Users/Magnifier/Desktop/data/machine_learning/cnn_chatbot/categories.pickle", 'wb') as f:
	pickle.dump(category, f)




