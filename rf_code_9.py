# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:32:04 2018

@author: Srinivas
"""

import pandas as pd
import numpy as np

traindata = pd.read_csv("train.csv",header = 0)
testdata = pd.read_csv("test.csv",header = 0)
sample= pd.read_csv("sample_submission.csv",header = 0)

import nltk
import re
import pickle
from nltk.corpus import stopwords

list_tweets = traindata["tweet"]
labels = traindata["label"]

clean_tweets = []

for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    words = nltk.word_tokenize(tweet)
    newwords = [word for word in words if word not in stopwords.words('english')]
    tweet = ' '.join(newwords)
    clean_tweets.append(tweet)

for i in range(len(clean_tweets)):
    clean_tweets[i] = re.sub(r"user"," ",clean_tweets[i])
    clean_tweets[i] = re.sub(r"^\s+","",clean_tweets[i])
    clean_tweets[i] = re.sub(r"ð","",clean_tweets[i])
    clean_tweets[i] = re.sub(r"\s+"," ",clean_tweets[i])
    
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

for i in range(len(clean_tweets)):
    words = nltk.word_tokenize(clean_tweets[i])
    newwords = [lemmatizer.lemmatize(word) for word in words]
    clean_tweets[i] = ' '.join(newwords)


from sklearn.feature_extraction.text import CountVectorizer

tfidf_vectorizer = CountVectorizer(max_features = 3000,
                                   min_df = 3, 
                                   max_df = 0.6, 
                                   stop_words = stopwords.words("english"))

X_tfidf = tfidf_vectorizer.fit_transform(clean_tweets).toarray()

import random
random.seed(500)

from sklearn.ensemble import RandomForestClassifier

reg = RandomForestClassifier(n_estimators = 1000,max_depth = 62,oob_score=True,random_state = 500,bootstrap = True)
reg.fit(X_tfidf,labels)
reg.oob_score_



new_tweets = testdata["tweet"]

prediction_tweets = []

for tweet in new_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    words = nltk.word_tokenize(tweet)
    newwords = [word for word in words if word not in stopwords.words('english')]
    tweet = ' '.join(newwords)
    prediction_tweets.append(tweet)
 
for i in range(len(prediction_tweets)):
    prediction_tweets[i] = re.sub(r"user"," ",prediction_tweets[i])
    prediction_tweets[i] = re.sub(r"^\s+","",prediction_tweets[i])
    prediction_tweets[i] = re.sub(r"\s+"," ",prediction_tweets[i])
    prediction_tweets[i] = re.sub(r"ð","",prediction_tweets[i])


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

for i in range(len(prediction_tweets)):
    words = nltk.word_tokenize(prediction_tweets[i])
    newwords = [lemmatizer.lemmatize(word) for word in words]
    prediction_tweets[i] = ' '.join(newwords)


tweet_tfidf = tfidf_vectorizer.transform(prediction_tweets)
tweet_tfidf = tweet_tfidf.toarray()
prediction = reg.predict(tweet_tfidf)
sample['label'] = prediction
print(sample["label"].value_counts())
sample.to_csv('prediction_file_9.csv',index=False)