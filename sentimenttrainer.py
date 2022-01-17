import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections
import random
import tweepy as tw
import nltk
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from testbot import find
import re, string
import networkx
import warnings
from timeit import default_timer as timer
import pickle
postweets = twitter_samples.tokenized('positive_tweets.json')
negtweets = twitter_samples.tokenized('negative_tweets.json')
stop_words = stopwords.words('english')
pos = []
neg = []
def removenoise(tokens,stop_words = ()):
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

for x in postweets:
    pos.append(removenoise(x,stop_words))
for x in negtweets:
    neg.append(removenoise(x,stop_words))

def get_words(clean):
    for tokens in clean:
        for token in tokens:
            yield token
all_pos = get_words(pos)
all_neg = get_words(neg)
posdist = FreqDist(all_pos)
negdist = FreqDist(all_neg)

def get_tweets(clean):
    for tokens in clean:
        yield dict([token,True] for token in tokens)

pos_toks = get_tweets(pos)
neg_toks = get_tweets(neg)

pos_data = [(tweet_dict, 'Positive') for tweet_dict in pos_toks]
neg_data = [(tweet_dict, 'Negative') for tweet_dict in neg_toks]

data_set = pos_data + neg_data

random.shuffle(data_set)

train = data_set[:7000]
test = data_set[7000:]

classifier = NaiveBayesClassifier.train(train)

#custom = "The saddest part about the #StimulusBill would be if they manage to wire the funds to people's accounts before Chri… https://t.co/aLH6K8Fghx"

#custom_token = removenoise(word_tokenize(custom))
start = timer()
alltweets = find('$SPY')
pcount = 0
ncount = 0
total = 0
for x in alltweets:
    custom_token = removenoise(word_tokenize(x))
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
















