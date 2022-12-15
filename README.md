# Project overview

* Scraped over 10 000 news article from glassdoor using the article search API of The New York Times.
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

# Code and Resources Used 

**Python Version :** 3.9.12                                                                 
**Packages :** pandas, numpy, tensorflow, requests, time, nltk, re, sklearn, matplotlib, seaborn, wordcloud, selenium, flask, json, pickle, random                                
**For Web Framework Requirements :**  ```pip install -r requirements.txt```                               
