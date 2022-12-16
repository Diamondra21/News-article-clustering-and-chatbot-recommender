# Project overview

* Scraped over 10 000 news article using the article search API from The New York Times.
* Pre-processed text data to cluster the articles with k-means algorithm. 
* Searched the optimal number of clusters with the elbow method.
* Built a chatbot to recommend an article, based on a user request.

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

The k-means method is used to cluster the data. The goal is to assign each point of the dataset (articles) into discrete groups. Where each point within the same cluster is close to each other and far from the points of different clusters.

To make this method work, its iterative algorithm is based on centroids (center points of each cluster). The number of clusters K is the hyperparameter that can be initialized randomly. But we’re using the algorithm K-Means++ algorithm which tends to converge more rapidly. 

# Code and Resources Used 

**Python Version :** 3.9.12                                                                 
**Packages :** pandas, numpy, tensorflow, requests, time, nltk, re, sklearn, matplotlib, seaborn, wordcloud, selenium, flask, json, pickle, random                                
**For Web Framework Requirements :**  ```pip install -r requirements.txt```                               
