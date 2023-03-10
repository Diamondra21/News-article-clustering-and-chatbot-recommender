# Project overview

* Scraped over 10 000 news article using the article search API from The New York Times.
* Pre-processed text data to cluster the articles with k-means algorithm. 
* Searched the optimal number of clusters with the elbow method.
* Built a chatbot to recommend an article, based on the user's article topic request.

# Web Scraping

Use of the article search API of The New York Times to scrape over 10 000 news article from nytimes.com.                                   
For each article, we got the following :
*	title
*	abstract
*	paragraph
*	date

# Data cleaning

The text data is preprocessing using the following usual NLP techniques :
* Special characters, numbers and punctuations are removed.
* We put each word in its lemmatized form.
* We remove stop words.

Lemmatizating is preferred to stemming for the better quality of clustering. Inertia is much more lower
when words are lemmatized.

After using the previous steps, the data is cleaned and optimized in order to be transformed into numerical data to be used effectively by machine learning models.

# K-means clustering 

The k-means model is used to cluster the data. The goal is to assign each point of the dataset (articles) into discrete groups. Where each point within the same cluster is close to each other and far from the points of different clusters.

In order to build this unsupervised model, its iterative algorithm is based on centroids (center points of each cluster).We used the k-means++ algorithm which tends to converge more rapidly. The number of clusters K is the hyperparameter we tuned with the elbow method. The ideal number of cluster we found is 20. 

As an unsupervised machine learning model, the k-means model we created will be used to cluser unseen data point based on the distance of the new data point and the centroids.

# Chatbot recommender

A chatbot is a software application used to conduct an on-line chat conversation via text. The chatbot works with a neural network to identify the patterns of sentences given by the user as input and pick a random response related to that query. The possible answers of the chatbot are predefined in an intents file, wich makes the chatbot highly customizable.

The chatbot will recommend articles to the user with the help of the k-means model created. The request related to the user's article topic (text) will be used as an entry point to be labelled by the k-means model. The recommended articles are 5 random articles of the dataset among the most recents, which have a cluster label corresponding to the prediction of the k-means model.

Here is an exemple of the use of the chatbot. The answers of the chatbot are in blue : 

<img src="chatbot_use_case.png" width=60% height=60%>

# Code and Resources Used 

**Python Version :** 3.9.12                                                                 
**Packages :** pandas, numpy, tensorflow, requests, time, nltk, re, sklearn, matplotlib, seaborn, wordcloud, selenium, flask, json, pickle, random                                
**For Web Framework Requirements :**  ```pip install -r requirements.txt```                               
