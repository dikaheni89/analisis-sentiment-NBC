# General:
#import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing
import csv

import time
#from selenium import webdriver
#from selenium.webdriver.common.keys import Keys
# For plotting and visualization:
from IPython.display import display
import matplotlib.pyplot as plt

import nltk
import re

file = 'testSentimen.csv'

# tentukan lokasi file, nama file, dan inisialisasi csv
f = open(file, 'r')
reader = csv.reader(f)

# membaca baris per baris
#for row in reader:
#	print (row)

# menutup file csv
f.close()
#print("DAFTAR TWEETS YANG TELAH DIAMBIL")

#initialize stopWords
stopWords = []

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#import regex
import re
#start process_tweet
def processTweet(tweet):
    # process the tweets
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end

#Read the tweets one by one and process it
fp = open('testSentimen.csv', encoding="utf8")
line = fp.readline()

stopWords = getStopWordList('stopwordsID.txt')

while line:
    processedTweet = processTweet(line)
    featureVector = getFeatureVector(processedTweet)
    #print (featureVector)
    line = fp.readline()
#end loop
fp.close()
print()
#print("DAFTAR STOPWORD")
print()

import array as arr
import csv

#Read the tweets one by one and process it
inpTweets = csv.reader(open('testSentimen.csv', encoding="utf8"), delimiter=',', quotechar='|')
tweets = []
featureList = []
for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet)
    tweets.append((featureVector, sentiment))
    featureList = featureList + featureVector
#end loop
#print("FEATURE LIST")
print()
#print (tweets)
#print (featureList)

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end

ini = processTweet('paket belum juga sampai')
ini2 = getFeatureVector(ini)
ini3 = extract_features(ini2)
#print (ini3)
pos = 0
neg = 0
neu = 0

#Read the tweets one by one and process it
fp = open('testSentimen.csv', encoding="utf8")
line = fp.readline()


stopWords = getStopWordList('stopwordsID.txt')

#end loop
#fp.close()
import nltk.classify
# Remove featureList duplicates
featureList = list(set(featureList))

# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)

# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

# Test the classifier
#testTweet = 'baru saja donor darah, tangan saya masih sakit'
testTweet = 'ujaran'
processedTestTweet = processTweet(testTweet)
sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
print ("testTweet = %s, sentiment = %s\n" % (testTweet, sentiment))
x = 0
sentimen=[]
while line:
    processedTweet = processTweet(line)
    featureVector = getFeatureVector(processedTweet)
#    #print (featureVector)
    line = fp.readline()
    testTweet = processedTweet
    processedTestTweet = processTweet(testTweet)
    sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
    sentimen.append(sentiment)
    print ("testTweet = %s, sentiment = %s\n" % (featureVector, sentiment))
    
    myData=featureVector
    x = x + 1
    if sentiment == 'positive':
        pos = 1 + pos
    elif sentiment == 'negative':
        neg = 1 + neg
    elif sentiment == 'neutral':
        neu = 1 + neu
        
print(x)
print()
print("ANALISIS SENTIMEN")
print()

print('Jumlah Sentiment Positive : ')
print(pos)
print('Jumlah Sentiment Negative : ')
print(neg)
print('Jumlah Sentiment Neutral :')
print(neu)

#Read the tweets one by one and process it
fp = open('testtopik.csv', encoding="utf8")
line = fp.readline()

stopWords = getStopWordList('stopwordsID.txt')

while line:
    processedTweet = processTweet(line)
    featureVector = getFeatureVector(processedTweet)
    #print (featureVector)
    line = fp.readline()
#end loop
fp.close()

import array as arr
import csv

#Read the tweets one by one and process it
inpTweets = csv.reader(open('testtopik.csv', encoding="utf8"), delimiter=',', quotechar='|')
tweets = []
featureList = []
for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet)
    tweets.append((featureVector, sentiment))
    featureList = featureList + featureVector
#end loop
#print (tweets)
#print (featureList)

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end

ini = processTweet('paket belum juga sampai')
ini2 = getFeatureVector(ini)
ini3 = extract_features(ini2)
print (ini3)
komentar = 0
layanan = 0
giat = 0
komentarpos = 0
komentarneg = 0
komentarneu = 0

layananpos = 0
layananneg = 0
layananneu = 0

giatpos = 0
giatneg = 0
giatneu = 0

#Read the tweets one by one and process it
fp = open('testSentimen.csv', encoding="utf8")
line = fp.readline()


#stopWords = getStopWordList('data/feature_list/stopwordsID.txt')

#end loop
#fp.close()
import nltk.classify
# Remove featureList duplicates
featureList = list(set(featureList))

# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)

# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

# Test the classifier
#testTweet = 'baru saja donor darah, tangan saya masih sakit'
testTweet = 'ujaran'
processedTestTweet = processTweet(testTweet)
sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
#print ("testTweet = %s, sentiment = %s\n" % (testTweet, sentiment))
x = 0
topik=[]
while line:
    processedTweet = processTweet(line)
    featureVector = getFeatureVector(processedTweet)
#    #print (featureVector)
    line = fp.readline()
    testTweet = processedTweet
    processedTestTweet = processTweet(testTweet)
    sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
    topik.append(sentiment)
    #print ("testTweet = %s, topik = %s, sentimen = %s\n" % (featureVector, sentiment,sentimen[x]))
    #print(sentimen[x])
    myData=featureVector
    x = x + 1
    if sentiment == 'KegiatanPengiriman':
        giat = 1 + giat
    elif sentiment == 'LayananPosIndonesia':
        layanan = 1 + layanan
    elif sentiment == 'KomentarMasyarakat':
        komentar = 1 + komentar
    
    
        
print()
print("KLASIFIKASI TOPIK")
print()
print(x)

print('Jumlah Kegiatan Pengiriman : ')
print(giat)
print('Jumlah Layanan Pos Indonesia : ')
print(layanan)
print('Jumlah Komentar Masyarakat :')
print(komentar)

for i in range(len(sentimen)):
    #print(i, sentimen[i])
    if topik[i] == 'KegiatanPengiriman' and sentimen[i] == 'positive':
        giatpos = 1 + giatpos
    elif topik[i] == 'KegiatanPengiriman' and sentimen[i] == 'negative':
        giatneg = 1 + giatneg
    elif topik[i] == 'KegiatanPengiriman' and sentimen[i] == 'neutral':
        giatneu = 1 + giatneu
    elif topik[i] == 'LayananPosIndonesia' and sentimen[i] == 'positive':
        layananpos = 1 + layananpos
    elif topik[i] == 'LayananPosIndonesia' and sentimen[i] == 'negative':
        layananneg = 1 + layananneg
    elif topik[i] == 'LayananPosIndonesia' and sentimen[i] == 'neutral':
        layananneu = 1 + layananneu
    elif topik[i] == 'KomentarMasyarakat' and sentimen[i] == 'positive':
        komentarpos = 1 + komentarpos
    elif topik[i] == 'KomentarMasyarakat' and sentimen[i] == 'negative':
        komentarneg = 1 + komentarneg
    elif topik[i] == 'KomentarMasyarakat' and sentimen[i] == 'neutral':
        komentarneu = 1 + komentarneu

    #else:
 #       print ("No match")
 
print ("****************\n")

print ("Layanan Pos Indonesia sentiment positive %s" % (layananpos))
print ("Layanan Pos Indonesia sentiment negative %s" % (layananneg))
print ("Layanan Pos Indonesia sentiment neutral %s" % (layananneu))
print ("\n")
print ("Kegiatan Pengiriman sentiment positive %s" % (giatpos))
print ("Kegiatan Pengiriman sentiment negative %s" % (giatneg))
print ("Kegiatan Pengiriman sentiment neutral %s" % (giatneu))
print ("\n")
print ("Komentar Masyarakat sentiment positive %s" % (komentarpos))
print ("Komentar Masyarakat sentiment negative %s" % (komentarneg))
print ("Komentar Masyarakat sentiment neutral %s" % (komentarneu))

objects = ('Negative', 'Neutral', 'Positive')
y_pos = np.arange(len(objects))
performance = [layananneg, layananneu, layananpos]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Jumlah dalam comments')
plt.title('Sentiment Topik Layanan Pos Indonesia')
 
plt.show()

import matplotlib.pyplot as plt
 
# Data to plot
print('Layanan Pos Indonesia')
labels = 'Positive', 'Negative', 'Neutral'
sizes = [layananpos, layananneg, layananneu]
colors = ['pink', 'slategray', 'royalblue']
explode = (0.2, 0.1, 0.1)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

# Data to plot
print('Komentar Masyarakat')
labels = 'Positive', 'Negative', 'Neutral'
sizes = [komentarpos, komentarneg, komentarneu]
colors = ['orangered', 'red', 'tomato']
explode = (0.1, 0.2, 0.1)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

# Data to plot
print('Kegiatan Pengiriman')
labels = 'Positive', 'Negative', 'Neutral'
sizes = [giatpos, giatneg, giatneu]
colors = ['tan', 'sienna', 'peru']
explode = (0.2, 0.1, 0.1)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

print('Analisis Sentimen')
labels = 'Positive', 'Negative', 'Neutral'
sizes = [pos, neg, neu]
colors = ['steelblue', 'slategray', 'gray']
explode = (0.2, 0.1, 0.1)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

print('Klasifikasi Topik')
labels = 'Kegiatan Pengiriman', 'Layanan Pos Indonesia', 'Komentar Masyarakat'
sizes = [giat, layanan, komentar]
colors = ['skyblue', 'lavender', 'pink']
explode = (0.2, 0.1, 0.1)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()