# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:51:55 2022

@author: USER
"""

import random
import json 
import pickle 
import numpy as np
import pandas as pd 

import nltk
from nltk.stem import WordNetLemmatizer 

from tensorflow.keras.models import load_model

df = pd.read_csv('NYT_articles_2022_clustered.csv',dtype={'article_id':str,'title':str,'abstract':str,'paragraph':str,'date':str,'text':str,'text_clean':str,'clusters':int}, parse_dates=['date'])

kmeans_model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_tfidf.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]),verbose = 0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]],'probabilitiy':str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents :
        if i['tag'] == tag :
            result = random.choice(i['responses'])
            break
    return result,tag

def get_article(request,vectorizer,kmeans_model):
    X = vectorizer.transform([request])
    cluster = kmeans_model.predict(X)[0]   
   
    df_articles = df[df['clusters']==cluster]
    
    titles = list(df_articles.nlargest(50, 'date')['title'])
    response = random.sample(titles,5)
    return '\n\n'.join(response)

#%%
print('Chatbot is running \n \n')            

# boolean used to end the conversation
disscussion = True

while disscussion == True :
    message = input("") # user text
    
    ints = predict_class(message) # prediction of the answer of the chatbot
    
    # answer of the chatbot and the tag related to the context of the answer
    res = get_response(ints, intents) 
    
    print('\n' + '\x1b[1;34m' + res[0] +'\x1b[0m') # printing the answer of the chatbot
    
    tag = res[1] # tag related to the context of the answer
    
    if tag == "article":
        request = input("")
        print('\n' +'\x1b[1;34m Here are the latest articles based on your request : \x1b[0m')
        print('\n' +'\x1b[1;34m' + get_article(request,vectorizer,kmeans_model) +'\x1b[0m')
    
    if tag == "goodbye":
        disscussion = False
            
print('\n \n Chatbot is off')            
            
            
            
            
            
            
            
            
            































