# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:49:56 2021

@author: chudc
"""

import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split


#STEP 1: load data
#list to select columns of interest
col_selection = ['text','Label']
corpus_true = pd.read_csv('True.csv', usecols = col_selection, dtype = {'text': np.str_, 'Label':np.int32})
corpus_false = pd.read_csv('Fake.csv', usecols = col_selection, dtype = {'text': np.str_, 'Label':np.int32})
#joined both df into one 
df = pd.concat([corpus_true,corpus_false], ignore_index = True)

#STEP 2: data pre-processing
#A) remove non alphabets
remove_non_alphabets = lambda x: re.sub(r'[^a-zA-Z]',' ',x)
#B)tokenn alphabets-only list
tokenize = lambda x: word_tokenize(x)
#C) remove stopwords
stopwords_1 = stopwords.words('english')
#only keep the words that not in nltk stopwords word list
bow_nostopwords = lambda x: [w for w in x if w not in stopwords_1]
#D)assign ps to a lambda function to run on each line of value
ps = PorterStemmer()
stem = lambda w: [ps.stem(x) for x in w]
#E)assign lemmatizer to a lambda function to run on each line of value
lemmatizer = WordNetLemmatizer()
leammtizer = lambda x: [lemmatizer.lemmatize(word) for word in x]

# apply all above methods to the column 'text'
df['text'] = df['text'].apply(remove_non_alphabets)
df['text'] = df['text'].apply(tokenize)
df['text'] = df['text'].apply(bow_nostopwords)
df['text'] = df['text'].apply(stem)
df['text'] = df['text'].apply(leammtizer)
df['text'] = df['text'].apply(lambda x: ' '.join(x))

#STEP 3: Split data into train and test
# split to 30 percent test data and 70 percent train data
train_corpus, test_corpus, train_labels, test_labels = train_test_split(df['text'], df['Label'], test_size=0.3)

#STEP 4: Construct Features for Machine Learning with different approaches
#A) Binary Feature Representation
# MIN_DF looks at how many documents contained a term, better known as document frequency
# ngram_range, use word level n-gram
# binary = True, then CountVectorizer no longer uses the counts of terms/tokens
#   If a token is present in a document, it is 1, if absent it is 0 regardless of its frequency of occurrence
binary_vectorizer=CountVectorizer(min_df=1, ngram_range=(1,1), binary = True)
binary_train_features = binary_vectorizer.fit_transform(train_corpus)
binary_test_features = binary_vectorizer.transform(test_corpus)

#B) Frequency Feature Representation
# Used CountVectorizer which uses the counts of terms/tokens
# MIN_DF looks at how many documents contained a term, better known as document frequency
# ngram_range, use word level n-gram
frequency_vectorizer=CountVectorizer(min_df=1, ngram_range=(1,1))
frequency_train_features = frequency_vectorizer.fit_transform(train_corpus)
frequency_test_features = frequency_vectorizer.transform(test_corpus)

#C) TFIDF Feature Representation
# Used TfidfVectorizer which converts a collection of raw documents to a matrix of TF-IDF features.
# MIN_DF looks at how many documents contained a term, better known as document frequency
# norm = 'l2': each output row will have unit norm. ‘l2’: Sum of squares of vector elements is 1. 
#   The cosine similarity between two vectors is their dot product when l2 norm has been applied. 
# smooth_idf = True: Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. 
#    Prevents zero divisions.
# use_idf= True: Enable inverse-document-frequency reweighting.
# ngram_range, use word level n-gram
tfidf_vectorizer=TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=(1,1))
tfidf_train_features = tfidf_vectorizer.fit_transform(train_corpus)  
tfidf_test_features = tfidf_vectorizer.transform(test_corpus)   

#D) BIGRAM Feature Representation
bigram_vectorizer=CountVectorizer(min_df=1, ngram_range=(2,2))
bigram_train_features = bigram_vectorizer.fit_transform(train_corpus)
bigram_test_features = bigram_vectorizer.transform(test_corpus)

#STEP 5: I'm using the user-defined functions from lab to train the models, perform predictions and evaluate them
# define a function to evaluate our classification models based on four metrics
# This defined function is also useful in other cases. This is comparing test_y and pred_y. 
# Both contain 1s and 0s.
def get_metrics(true_labels, predicted_labels):
    metrics_dict = dict(zip(["accuracy", "precision", "recall", "f1"], [None]*4))
    #metrics_dict = {i:None for i in ["accuracy", "precision", "recall", "f1"]}
    for m in metrics_dict.keys():
        exec('''metrics_dict['{}'] = np.round(                                                    
                        metrics.{}_score(true_labels, 
                                               predicted_labels),
                        2)'''.format(m, m))
    return metrics_dict

# define a function that trains the model, performs predictions and evaluates the predictions
def train_predict_evaluate_model(classifier, 
                                 train_features, train_labels, 
                                 test_features, test_labels):
    # build model    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    # evaluate model prediction performance   
    '''get_metrics(true_labels=test_labels, 
                predicted_labels=predictions)'''
    print(metrics.classification_report(test_labels,predictions))
    print(metrics.confusion_matrix(test_labels, predictions)) #included this line to the function to get the confusion matrix
    return predictions, get_metrics(true_labels=test_labels, predicted_labels=predictions)

#STEP 6: Classifier
# Support Vector Machine
# assign support vector machine function to an object
# C: Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
# kernel = 'linear': Specifies the kernel type to be used in the algorithm
# gamma = 'auto': 
SVM_binary = svm.SVC(C=1.0, kernel='linear')
SVM_frequency = svm.SVC(C=1.0, kernel='linear')
SVM_tfidf = svm.SVC(C=1.0, kernel='linear')
SVM_bigram = svm.SVC(C=1.0, kernel='linear')
#1. Predict and evaluate classifier with binary feature representation
SVM_binary_predictions, SVM_binary_metrics= train_predict_evaluate_model(classifier=SVM_binary,
                                           train_features=binary_train_features,
                                           train_labels=train_labels,
                                           test_features=binary_test_features,
                                           test_labels=test_labels)

#2. Predict and evaluate classifier with frequency feature representation
SVM_frequency_predictions, SVM_frequency_metrics = train_predict_evaluate_model(classifier=SVM_frequency,
                                           train_features=frequency_train_features,
                                           train_labels=train_labels,
                                           test_features=frequency_test_features,
                                           test_labels=test_labels)

#3. Predict and evaluate classifier with TFIDF feature representation
SVM_tfidf_predictions, SVM_tfidf_metrics = train_predict_evaluate_model(classifier=SVM_tfidf,
                                           train_features=tfidf_train_features,
                                           train_labels=train_labels,
                                           test_features=tfidf_test_features,
                                           test_labels=test_labels)

#4. Predict and evaluate classifier with TFIDF feature representation
SVM_bigram_predictions, SVM_bigram_metrics = train_predict_evaluate_model(classifier=SVM_bigram,
                                           train_features=bigram_train_features,
                                           train_labels=train_labels,
                                           test_features=bigram_test_features,
                                           test_labels=test_labels)
#STEP 7: Classify new data
#STEP 1: load data
newdata_true = pd.read_csv('Newdata_true.csv', usecols = col_selection, dtype = {'text': np.str_, 'Label':np.int32})
newdata_false = pd.read_csv('Newdata_fake.csv', usecols = col_selection, dtype = {'text': np.str_, 'Label':np.int32})
#joined both df into one 
df_newdata = pd.concat([newdata_true,newdata_false], ignore_index = True)

#STEP 2: data pre-processing
# apply all above methods to the column 'MESSAGE'
df_newdata['text'] = df_newdata['text'].apply(remove_non_alphabets)
df_newdata['text'] = df_newdata['text'].apply(tokenize)
df_newdata['text'] = df_newdata['text'].apply(bow_nostopwords)
df_newdata['text'] = df_newdata['text'].apply(stem)
df_newdata['text'] = df_newdata['text'].apply(leammtizer)
df_newdata['text'] = df_newdata['text'].apply(lambda x: ' '.join(x))

#only text articles
df_newdata_text = df_newdata['text']

#STEP 4: Construct Features for Machine Learning with different approaches
#A) Binary Feature Representation
binary_newdata = binary_vectorizer.transform(df_newdata_text)

#B) Frequency Feature Representation
frequency_newdata = frequency_vectorizer.transform(df_newdata_text)

#C) TFIDF Feature Representation   
tfidf_newdata = tfidf_vectorizer.transform(df_newdata_text)  

#C) TFIDF Feature Representation   
bigram_newdata = bigram_vectorizer.transform(df_newdata_text)  

#predict class using SVM and the different approaches
classifier_output_binary = SVM_binary.predict(binary_newdata)
classifier_output_frequency = SVM_frequency.predict(frequency_newdata)
classifier_output_tfidf = SVM_tfidf.predict(tfidf_newdata)
classifier_output_bigram = SVM_bigram.predict(bigram_newdata)

#confusion matrix
print(metrics.confusion_matrix(df_newdata['Label'], classifier_output_binary))
print(metrics.accuracy_score(df_newdata['Label'], classifier_output_binary)*100)
print(metrics.confusion_matrix(df_newdata['Label'], classifier_output_frequency))
print(metrics.accuracy_score(df_newdata['Label'], classifier_output_frequency)*100)
print(metrics.confusion_matrix(df_newdata['Label'], classifier_output_tfidf))
print(metrics.accuracy_score(df_newdata['Label'], classifier_output_tfidf)*100)
print(metrics.confusion_matrix(df_newdata['Label'], classifier_output_bigram))
print(metrics.accuracy_score(df_newdata['Label'], classifier_output_bigram)*100)

# add three columns with the predictions
df_newdata['Binary Prediction'] = classifier_output_binary.tolist()
df_newdata['Frequency Prediction'] = classifier_output_frequency.tolist()
df_newdata['TFIDF Prediction'] =  classifier_output_tfidf.tolist()
df_newdata['Bigram Prediction'] =  classifier_output_bigram.tolist()

# save predictions in csv file
df_newdata.to_csv('SVM_predictions.csv', index = False)







