# Project overview

* Created a tool that estimates data science salaries (MAE ~ $ 11K) to help data scientists negotiate their income when they get a job.
* Scraped over 1000 job descriptions from glassdoor using python and selenium
* Engineered features from the text of each job description to quantify the value companies put on python, excel, aws, and spark. 
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model. 
* Built a client facing API using flask 

# Web Scraping

Use of the article search API of The New York Times to scrape over 10 000 news article from nytimes.com. For each article, we got the following :
*	title
*	abstract
*	paragraph
*	date

# Code and Resources Used 

**Python Version :** 3.9.12                                                                 
**Packages :** pandas, numpy, tensorflow, requests, time, nltk, re, sklearn, matplotlib, seaborn, wordcloud, selenium, flask, json, pickle, random                                
**For Web Framework Requirements :**  ```pip install -r requirements.txt```                               
