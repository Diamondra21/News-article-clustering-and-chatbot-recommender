# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:15:32 2022

@author: USER
"""
import json
import nltk
from nltk.stem import WordNetLemmatizer 
import numpy as np
import random
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read()) #loading the intents file

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

# collecting all the intents components 
for intent in intents['intents']:
    for pattern in intent['patterns']:
        words_list = nltk.word_tokenize(pattern)
        words.extend(words_list)
        documents.append((words_list, intent['tag']))
        
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# lemmatizing each word collected
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

# sorting the words and removing duplicates
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training = []
output_empty = [0]*len(classes)

for document in documents :
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row])

random.shuffle(training)
training = np.array(training)

X_train = list(training[:,0])
y_train = list(training[:,1])

model = Sequential()
model.add(Dense(128,input_shape = (len(X_train[0]),),activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation = 'softmax'))

sgd = SGD(learning_rate = 0.01,momentum = 0.9,nesterov=True)
model.compile(loss = 'categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

hist = model.fit(np.array(X_train),np.array(y_train), epochs=200, batch_size=5,verbose=1)
model.save('chatbot_model.h5',hist)

































