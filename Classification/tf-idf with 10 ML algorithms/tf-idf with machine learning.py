# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:16:40 2022

@author: smrya
"""
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
np.random.seed(0)
#%% Load original texts and labels
df=pd.read_json(r'C:\Users\smrya\Desktop\All_Beauty_5.json', lines=True)
texts = [' '.join([str(i),str(j)]) for i,j in zip(df['reviewText'],df['summary'])]
labels = [i for i in df['overall']]

# Classes are imbalanced, so we need to remove some samples. 
new_texts=[]
new_labels=[]
N_5=0
for i in range(len(labels)):
  if labels[i]==5:
    N_5+=1
    if N_5<156:new_texts.append(texts[i]);new_labels.append(labels[i])
  else: new_texts.append(texts[i]);new_labels.append(labels[i])

# Assign new texts and labels as our dataset
texts=new_texts
labels=new_labels
#%% clean the text: choose as you need
# 1. remove blanks rows if any; 2. change all the text in to lower case; 3. word tokenization
# 4. remove stop words; 5. remove non-alpha text; 6. word lemmatization/stemming (differences: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34) 

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

#lower case
texts=[i.lower() for i in texts]

#tokenize
texts=[word_tokenize(i) for i in texts]

#stemming/lemmatization
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
lemmatized_texts=[]
for i in range(len(texts)):
    newwords = []# Declaring Empty List to store the words that follow the rules for this step
    word_lemmatized=WordNetLemmatizer()#initialize lemmatizer model
    for word, tag in pos_tag(texts[i]):
        if word not in stopwords.words('english') and word.isalpha(): #isalpha removes punctuations
            word=word_lemmatized.lemmatize(word,tag_map[tag[0]])
            newwords.append(word)
    lemmatized_texts.append(str(newwords)) #[m,text]
#%% transform the labels to 0,1,2,3,4,5
target_names = list(set(labels))
label2idx = {label: idx for idx, label in enumerate(target_names)}
labels=[label2idx[i] for i in labels]
#%% split the data into train and test
from sklearn import model_selection
train_text, test_text, train_label, test_label = model_selection.train_test_split(lemmatized_texts,labels,test_size=0.25)
#%% create TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf_vect = TfidfVectorizer(max_features=2698)
Tfidf_vect.fit(lemmatized_texts)
train_text_idf = Tfidf_vect.transform(train_text)
test_text_idf = Tfidf_vect.transform(test_text)
len(Tfidf_vect.vocabulary_) #look up the number of words in the vocab
#%% Naive Bayes 
from sklearn import naive_bayes
# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(train_text_idf,train_label)
# predict the labels on validation dataset
predictions_NB = Naive.predict(test_text_idf)
# Use accuracy_score function to get the accuracy
acc_NB=accuracy_score(predictions_NB, test_label)*100
print (acc_NB)
#%% SVM
from sklearn import svm
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(train_text_idf,train_label)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(test_text_idf)
# Use accuracy_score function to get the accuracy
acc_SVM=accuracy_score(predictions_SVM, test_label)*100
print (acc_SVM)
#%% Logistic regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(train_text_idf,train_label)
predictions_LR=LR.predict(test_text_idf)
# Use accuracy_score function to get the accuracy
acc_LR=accuracy_score(predictions_LR, test_label)*100
print (acc_LR)
#%% RF
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(train_text_idf,train_label)
predictions_RF=RF.predict(test_text_idf)
# Use accuracy_score function to get the accuracy
acc_RF=accuracy_score(predictions_RF, test_label)*100
print (acc_RF)
#%% K-NN
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=1)
KNN.fit(train_text_idf,train_label)
predictions_KNN=KNN.predict(test_text_idf)
# Use accuracy_score function to get the accuracy
acc_KNN=accuracy_score(predictions_KNN, test_label)*100
print (acc_KNN)
#%% Gradient Boosting DT
from sklearn.ensemble import GradientBoostingClassifier
GBDT=GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, max_depth=5, random_state=0)
GBDT.fit(train_text_idf,train_label)
predictions_GBDT=GBDT.predict(test_text_idf)
# Use accuracy_score function to get the accuracy
acc_GBDT=accuracy_score(predictions_GBDT, test_label)*100
print (acc_GBDT)
#%% XGBOOST
#from xgboost.sklearn import XGBClassifier
import xgboost
xgb= xgboost.XGBClassifier(objective="multi:softprob", eta=0.25, max_depth=31)
xgb.fit(train_text_idf,train_label)
predictions_xgb=xgb.predict(test_text_idf)
# Use accuracy_score function to get the accuracy
acc_xgb=accuracy_score(predictions_xgb, test_label)*100
print (acc_xgb)
#%% SGDClassifier
from sklearn.linear_model import SGDClassifier
SGD=SGDClassifier()
SGD.fit(train_text_idf,train_label)
predictions_SGD=SGD.predict(test_text_idf)
# Use accuracy_score function to get the accuracy
acc_SGD=accuracy_score(predictions_SGD, test_label)*100
print (acc_SGD)
#%% DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(train_text_idf,train_label)
predictions_DT=DT.predict(test_text_idf)
# Use accuracy_score function to get the accuracy
acc_DT=accuracy_score(predictions_DT, test_label)*100
print (acc_DT)
#%% LGBMClassifier
from lightgbm import LGBMClassifier
LGBM=LGBMClassifier()
LGBM.fit(train_text_idf,train_label)
predictions_LGBM=LGBM.predict(test_text_idf)
# Use accuracy_score function to get the accuracy
acc_LGBM=accuracy_score(predictions_LGBM, test_label)*100
print (acc_LGBM)
#%% Results of all ML algorithms
keys=['LGBM', 'DT', 'SGD', 'xgb', 'GBDT', 'KNN', 'RF', 'LR', 'SVM', 'NB']
values=[acc_LGBM, acc_DT, acc_SGD, acc_xgb, acc_GBDT, acc_KNN, acc_RF, acc_LR, acc_SVM, acc_NB]
accuracies=dict(zip(keys,values))
accuracies=dict(sorted(accuracies.items(), key=lambda item: item[1]))
accuracies
