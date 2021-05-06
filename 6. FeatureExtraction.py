# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:16:43 2021

@author: chudc
"""

import pandas as pd
import nltk 
from nltk import FreqDist
from nltk.corpus import stopwords

#read file articles classified by SVM model and Bigram text representation
#list to select columns of interest
all_newdata = pd.read_csv('Classified_data.csv')
#create two df - one for true and one for fake
classified_true = all_newdata[all_newdata['Bigram Prediction'] == 1]
classified_fake = all_newdata[all_newdata['Bigram Prediction'] == 0]

#articles to list without label
list_true = classified_true['text'].values.tolist()
list_fake = classified_fake['text'].values.tolist()

#1) Use a simple bag-of-words approach
#user defined function to tokenize the elements of each of the articles
def bow(x):
    tokens = []
    for i in range(len(x)):
        tokens.append(nltk.word_tokenize(x[i]))
        
    return tokens

#use function to obtain tokens
tokenized_true = bow(list_true)
tokenized_fake = bow(list_fake)
#merge tokens in one list
list_tokens_true = [j for i in tokenized_true for j in i]
list_tokens_fake = [j for i in tokenized_fake for j in i]
#change all tokens into lower case  and only keep text words
words_true = [w.lower() for w in list_tokens_true if w.isalpha()]   
words_fake = [w.lower() for w in list_tokens_fake if w.isalpha()]   
#generate a frequency dictionary for all tokens 
freqbow_true = FreqDist(words_true)
freqbow_fake = FreqDist(words_fake)
#sort the frequency list in descending order
sortedfreqbow_true = sorted(freqbow_true.items(),key = lambda k:k[1], reverse = True)
sortedfreqbow_fake = sorted(freqbow_fake.items(),key = lambda k:k[1], reverse = True)
#print top 50 terms
print(sortedfreqbow_true[0:50])
print(sortedfreqbow_fake[0:50])
#plot the top 50 terms
freqbow_true.plot(30)
freqbow_fake.plot(30)

#2) Use a bag-of-words approach with stemming and stop words removal 
#import NLTK stopwords
stopwords1 = stopwords.words('english')
# #decided to remove some additional ones to improve the results
# add_words = ['news', 'january','press','finance',
#                  'copyright','llc','inc','report', 'article']
# stopwords.extend(add_words)
#only keep the words that not in nltk stopwords word list
bow_nostopwords_true = [w for w in words_true if w not in stopwords1]
bow_nostopwords_fake = [w for w in words_fake if w not in stopwords1]

#Decided to first try a Stemmer: Porter
porter = nltk.PorterStemmer()
bow_pstem_true = [porter.stem(w) for w in bow_nostopwords_true]
bow_pstem_fake = [porter.stem(w) for w in bow_nostopwords_fake]

#generate a frequency dictionary for all tokens 
freqbow_nostw_pstem_true = FreqDist(bow_pstem_true)
freqbow_nostw_pstem_fake = FreqDist(bow_pstem_fake)
#sort the frequency list in decending order
sortedfreqbow_nostw_pstem_true = sorted(freqbow_nostw_pstem_true.items(),key = lambda k:k[1], reverse = True)
sortedfreqbow_nostw_pstem_fake = sorted(freqbow_nostw_pstem_fake.items(),key = lambda k:k[1], reverse = True)
#print top 50 terms
print(sortedfreqbow_nostw_pstem_true[0:50])
print(sortedfreqbow_nostw_pstem_fake[0:50])
#plot top 50 terms
freqbow_nostw_pstem_true.plot(30)
freqbow_nostw_pstem_fake.plot(30)

#Now I will try a Lemmatizer: WordNet
#Steps are the same with the previous two stemmers 
lemma = nltk.WordNetLemmatizer()
bow_wordnet_true = [lemma.lemmatize(w) for w in bow_nostopwords_true]
bow_wordnet_fake = [lemma.lemmatize(w) for w in bow_nostopwords_fake]
#generate a frequency dictionary for all tokens 
freqbot_nostw_wordnet_true = FreqDist(bow_wordnet_true)
freqbot_nostw_wordnet_fake = FreqDist(bow_wordnet_fake)
#sort the frequency list in decending order
sortedfreqbow_nostw_wordnet_true = sorted(freqbot_nostw_wordnet_true.items(),key = lambda k: k[1], reverse = True)
sortedfreqbow_nostw_wordnet_fake = sorted(freqbot_nostw_wordnet_fake.items(),key = lambda k: k[1], reverse = True)
#print top 50 terms
print(sortedfreqbow_nostw_wordnet_true[0:50])
print(sortedfreqbow_nostw_wordnet_fake[0:50])
#plot top 50 terms
freqbot_nostw_wordnet_true.plot(30)
freqbot_nostw_wordnet_fake.plot(30)

#3)	Use POS approach and focus on all the noun forms (NN, NNP, NNS, NNPS)
#use unprocessed tokens, keep capitalization, punctuation and stopwords
POS_tags_true = nltk.pos_tag(list_tokens_true)
POS_tags_fake = nltk.pos_tag(list_tokens_fake)
#Generate a list of POS tags and remove special characters as suggested in Piazza
POS_tag_list_true = [(word,tag) for (word,tag) in POS_tags_true if tag.startswith('N') and word[0].isalnum()]
POS_tag_list_fake = [(word,tag) for (word,tag) in POS_tags_fake if tag.startswith('N') and word[0].isalnum()]
#Generate a frequency distribution of all the POS tags
tag_freq_true = nltk.FreqDist(POS_tag_list_true)
tag_freq_fake = nltk.FreqDist(POS_tag_list_fake)
#Sort the result 
sorted_tag_freq_true = sorted(tag_freq_true.items(), key = lambda k:k[1], reverse = True)
sorted_tag_freq_fake = sorted(tag_freq_fake.items(), key = lambda k:k[1], reverse = True)
#print top 50 terms
print(sorted_tag_freq_true[0:50])
print(sorted_tag_freq_fake[0:50])
#plot top 50 terms
tag_freq_true.plot(30)
tag_freq_fake.plot(30)

#4)	Use POS approach and only focus on NNP 
POS_tag_list_NNP_true = [(word,tag) for (word,tag) in POS_tags_true if tag == 'NNP' and word[0].isalnum()]
POS_tag_list_NNP_fake = [(word,tag) for (word,tag) in POS_tags_fake if tag == 'NNP' and word[0].isalnum()]
#Generate a frequency distribution of all the POS tags
tag_freq_NNP_true = nltk.FreqDist(POS_tag_list_NNP_true)
tag_freq_NNP_fake = nltk.FreqDist(POS_tag_list_NNP_fake)
#Sort the result 
sorted_tag_freq_NNP_true = sorted(tag_freq_NNP_true.items(), key = lambda k:k[1], reverse = True)
sorted_tag_freq_NNP_fake = sorted(tag_freq_NNP_fake.items(), key = lambda k:k[1], reverse = True)
#print top 50 terms
print(sorted_tag_freq_NNP_true[0:50])
print(sorted_tag_freq_NNP_fake[0:50])
#plot top 30 terms
tag_freq_NNP_true.plot(30)
tag_freq_NNP_fake.plot(30)









