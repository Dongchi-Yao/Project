# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:36:49 2022

@author: smrya
"""

#%% 
# Note 1: 
    # This is for transforming a list of raw sentences [m,1] to [m, words]
    # word_tokenization; lower case; remove stop words; remove non-alpha; word lemmatization
# Note 2:
    # More suitable for non-BERT task
# Note 3:
    # The result can be used for creating tf-idf/wordcount or word-2-vec static embedding

#clean the text: choose as you need
# 1. remove blanks rows if any; 2. change all the text in to lower case; 3. word tokenization
# 4. remove stop words; 5. remove non-alpha text; 6. word lemmatization/stemming (differences: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34) 

def Raw2Lemmatized(texts):
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet as wn
    import nltk
    from nltk.corpus import wordnet as wn
    from collections import defaultdict
    from tqdm import tqdm
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('omw-1.4')
    
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
    for i in tqdm(range(len(texts))):
        newwords = []# Declaring Empty List to store the words that follow the rules for this step
        word_lemmatized=WordNetLemmatizer()#initialize lemmatizer model
        for word, tag in pos_tag(texts[i]):
            if word not in stopwords.words('english') and word.isalpha(): #isalpha removes punctuations
                word=word_lemmatized.lemmatize(word,tag_map[tag[0]])
                newwords.append(word)
        newwords=(' ').join(newwords)
        lemmatized_texts.append(newwords) #[m,text]
    return lemmatized_texts