# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:20:07 2021

@author: chudc
"""

import pandas as pd
import gensim
from gensim import corpora,models
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import re
from pprint import pprint
from gensim.models import LsiModel

#read file articles classified by SVM model and Bigram text representation
#list to select columns of interest
all_newdata = pd.read_csv('Classified_data.csv')
#create two df - one for true and one for fake
classified_true = all_newdata[all_newdata['Bigram Prediction'] == 1]
classified_fake = all_newdata[all_newdata['Bigram Prediction'] == 0]

#convert all articles into lists
list_true = classified_true['text'].tolist()
list_fake = classified_fake['text'].tolist()

# Tokenize the articles
# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(list_true)):
    list_true[idx] = list_true[idx].lower()  # Convert to lowercase.
    list_true[idx] = tokenizer.tokenize(list_true[idx])  # Split into words.
for idx in range(len(list_fake)):
    list_fake[idx] = list_fake[idx].lower()  # Convert to lowercase.
    list_fake[idx] = tokenizer.tokenize(list_fake[idx])  # Split into words.

# Remove numbers, but not words that contain numbers.
list_true1 = [[token for token in doc if not token.isnumeric()] for doc in list_true]
list_fake1 = [[token for token in doc if not token.isnumeric()] for doc in list_fake]
    
# Remove stopwords.
list_true2 = [[token for token in doc if token not in stopwords.words('english')] for doc in list_true1]
list_fake2 = [[token for token in doc if token not in stopwords.words('english')] for doc in list_fake1]

# Remove words that are only one character.
list_true3 = [[token for token in doc if len(token) > 1] for doc in list_true2]
list_fake3 = [[token for token in doc if len(token) > 1] for doc in list_fake2]

# Lemmatize the documents.
lemmatizer = WordNetLemmatizer()
list_true4 = [[lemmatizer.lemmatize(token) for token in doc] for doc in list_true3]
list_fake4 = [[lemmatizer.lemmatize(token) for token in doc] for doc in list_fake3]

# Compute bigrams
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram_true = Phrases(list_true4, min_count=10)
for idx in range(len(list_true4)):
    for token in bigram_true[list_true4[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            list_true4[idx].append(token)
bigram_fake = Phrases(list_fake4, min_count=10)
for idx in range(len(list_fake4)):
    for token in bigram_fake[list_fake4[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            list_fake4[idx].append(token)

# Remove rare and common tokens
# Create a dictionary representation of the documents.
dictionary_true = Dictionary(list_true4)
dictionary_fake = Dictionary(list_fake4)


#GENERATE TERM DOCUMENT MATRIX
# Bag-of-words representation of the documents.
corpus_true = [dictionary_true.doc2bow(doc) for doc in list_true4]
corpus_fake = [dictionary_fake.doc2bow(doc) for doc in list_fake4]
print('True Articles - Number of unique tokens: %d' % len(dictionary_true))
print('Fake Articles - Number of unique tokens: %d' % len(dictionary_fake))
print('True Articles - Number of documents: %d' % len(corpus_true))
print('Fake Articles - Number of documents: %d' % len(corpus_fake))

# generate a unique token list 
sort_token_true = sorted(dictionary_true.items(),key=lambda k:k[0], reverse = False)
sort_token_fake = sorted(dictionary_fake.items(),key=lambda k:k[0], reverse = False)
unique_token_true = [token.encode('utf8') for (ID,token) in sort_token_true]
unique_token_fake = [token.encode('utf8') for (ID,token) in sort_token_fake]

# create matrix
matrix_true = gensim.matutils.corpus2dense(corpus_true,num_terms=len(dictionary_true),dtype = 'int')
matrix_fake = gensim.matutils.corpus2dense(corpus_fake,num_terms=len(dictionary_fake),dtype = 'int')
#transpose the matrix 
matrix_true = matrix_true.T 
matrix_fake = matrix_fake.T 

#convert the numpy matrix into pandas data frame
matrix_df_true = pd.DataFrame(matrix_true, columns=unique_token_true)
matrix_df_fake = pd.DataFrame(matrix_fake, columns=unique_token_fake)

#write matrix dataframe into csv
matrix_df_true.to_csv('Term_Document_matrix_True.csv')
matrix_df_fake.to_csv('Term_Document_matrix_Fake.csv')

#LDA MODEL
# Train LDA model
# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 20
iterations = 100
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp_true = dictionary_true[0]  # This is only to "load" the dictionary
temp_fake = dictionary_fake[0] 
id2word_true = dictionary_true.id2token
id2word_fake = dictionary_fake.id2token

lda_true = LdaModel(corpus=corpus_true, id2word=id2word_true,
    chunksize=chunksize, alpha='auto', eta='auto', iterations=iterations,
    num_topics=num_topics, passes=passes, eval_every=eval_every)

lda_fake = LdaModel(corpus=corpus_fake, id2word=id2word_fake,
    chunksize=chunksize, alpha='auto', eta='auto', iterations=iterations,
    num_topics=num_topics, passes=passes, eval_every=eval_every)

lda_true.print_topics(10) #V matrix, topic matrix
lda_fake.print_topics(10) #V matrix, topic matrix

for i,topic in lda_true.print_topics(10):
    print(f'True Articles - Top 10 words for topic #{i+1}:')
    print(",".join(re.findall('".*?"',topic)))
    print('\n')

for i,topic in lda_fake.print_topics(10):
    print(f'Fake Articles - Top 10 words for topic #{i+1}:')
    print(",".join(re.findall('".*?"',topic)))
    print('\n')

#top topics for true and fake articles
top_topics_true = lda_true.top_topics(corpus_true) 
top_topics_fake = lda_fake.top_topics(corpus_fake) 

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence_true = sum([t[1] for t in top_topics_true]) / num_topics
avg_topic_coherence_fake = sum([t[1] for t in top_topics_fake]) / num_topics
print('True Articles - Average topic coherence: %.4f.' % avg_topic_coherence_true)
print('Fake Articles - Average topic coherence: %.4f.' % avg_topic_coherence_fake)

pprint(top_topics_true)
pprint(top_topics_fake)

# Generate U Matrix for LDA model for true and fake
corpus_lda_true = lda_true[corpus_true] #transform lda model
corpus_lda_fake = lda_fake[corpus_fake]

#convert corpus_lda to numpy matrix
U_matrix_lda_true = gensim.matutils.corpus2dense(corpus_lda_true,num_terms=10).T
U_matrix_lda_fake = gensim.matutils.corpus2dense(corpus_lda_fake,num_terms=10).T

#write U_matrix into pandas dataframe and output
U_matrix_lda_df_true = pd.DataFrame(U_matrix_lda_true)
U_matrix_lda_df_true.to_csv('U_matrix_lda_True.csv')
U_matrix_lda_df_fake = pd.DataFrame(U_matrix_lda_fake)
U_matrix_lda_df_fake.to_csv('U_matrix_lda_Fake.csv')

#print shape of the matrix
print('Term-Doc Matrix for True Articles:', matrix_df_true.shape)
print('Term-Doc Matrix for Fake Articles:',matrix_df_fake.shape)
print('U Matrix for True Articles:', U_matrix_lda_df_true.shape)
print('U Matrix for Fake Articles:',U_matrix_lda_df_fake.shape)

# LSI MODEL
# Tfidf Transformation 
tfidf_true = models.TfidfModel(corpus_true) #fit tfidf model true articles
tfidf_fake = models.TfidfModel(corpus_fake) #fit tfidf model fake articles
#transform tfidf model 
corpus_tfidf_true = tfidf_true[corpus_true]
corpus_tfidf_fake = tfidf_fake[corpus_fake]    

# Train LSI model.
lsi_true = models.LsiModel(corpus_tfidf_true, id2word=dictionary_true, num_topics=10)
lsi_fake = models.LsiModel(corpus_tfidf_fake, id2word=dictionary_fake, num_topics=10)

#Generate topics for true articles
for i,topic in lsi_true.print_topics(10):
    print(f'True Articles - Top 10 words for topic #{i+1}:')
    print(",".join(re.findall('".*?"',topic)))
    print('\n')

#Generate topics for fake articles
for i,topic in lsi_fake.print_topics(10):
    print(f'Fake Articles - Top 10 words for topic #{i+1}:')
    print(",".join(re.findall('".*?"',topic)))
    print('\n')
    
# Generate U Matrix for LSI model
corpus_lsi_true = lsi_true[corpus_tfidf_true] #transform lda model
corpus_lsi_fake = lsi_fake[corpus_tfidf_fake] #transform lda model

#convert corpus_lsi to numpy matrix
U_matrix_lsi_true = gensim.matutils.corpus2dense(corpus_lsi_true,num_terms=10).T
U_matrix_lsi_fake = gensim.matutils.corpus2dense(corpus_lsi_fake,num_terms=10).T

#write U_matrix into pandas dataframe and output
pd.DataFrame(U_matrix_lsi_true).to_csv('U_matrix_lsi_True.csv')
pd.DataFrame(U_matrix_lsi_fake).to_csv('U_matrix_lsi_Fake.csv')



