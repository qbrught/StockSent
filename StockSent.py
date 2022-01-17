import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections
import random
import tweepy as tw
import nltk
import PySimpleGUI as sg
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import re, string
import networkx
import warnings
import datetime
import warnings
from timeit import default_timer as timer
import pickle
class StockSent:
    def removenoise(self,tokens,stop_words = ()):
        lemmatizer = WordNetLemmatizer()
        clean = []
        for token, tag in pos_tag(tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)

            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            token = lemmatizer.lemmatize(token,pos)
            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                clean.append(token.lower())
        return clean
    
    def find(self,symbol,twitems):
        c_key='xUKWKE41XcNdzPxFE6vOxSVoZ'
        c_sec='VFQtZYC79nfbexe1MnppgQSLfisHGmKUmwNJi06xrC8yYTMyzf'
        atk='1337141195307282432-ngOe63wUIYpqEWRb0xBXJFlN4j8CpN'
        ats='rhUc5L9kqXYZKz9AAEtar9mVJoPGaGCQDOBh3rbR4S96O'

        auth = tw.OAuthHandler(c_key, c_sec)
        auth.set_access_token(atk, ats)
        api = tw.API(auth, wait_on_rate_limit=True)
                  
                     
        search_words = symbol + "-filter:retweets"
        date_since = datetime.date.today()

        tweets = tw.Cursor(api.search,
                        q= search_words,
                        lang = 'en',
                        since = date_since).items(twitems)
        alltweets = [tweet.text.replace('\n','') for tweet in tweets]
        return alltweets
def main():
    sg.theme('DarkAmber')
    layout = [ [sg.Text("Symbol"), sg.InputText()],
                [sg.Text("Total Tweets"), sg.InputText()],
                [sg.OK(), sg.Cancel()]]

    window = sg.Window("StockSentiment", layout)
    s1 = StockSent()
    while True:
        event,values = window.read()
        if event in (sg.WIN_CLOSED, "Cancel"):
            break
        if event in (sg.WIN_CLOSED, "OK"):
            symbol = str(values[0])
            total_items = int(values[1])
            classifier_f = open('stock.pickle','rb')
            classifier = pickle.load(classifier_f)

            start = timer()
            alltweets = s1.find('$'+symbol,int(total_items))
            pcount = 0
            ncount = 0
            total = 0
            for x in alltweets:
                custom_token = s1.removenoise(word_tokenize(x))
                if classifier.classify(dict([token,True] for token in custom_token)) == 'Positive':
                    pcount += 1
                    total += 1
                elif classifier.classify(dict([token,True] for token in custom_token)) == 'Negative':
                    ncount += 1
                    total += 1

            if pcount > ncount:
                print('Positive:', pcount/total)
            elif ncount > pcount:
                print('Negative:', ncount/total)
            else:
                print('neutral')
            end = timer()
            print(end-start)
            classifier_f.close()

if __name__ == "__main__":  
    main()

