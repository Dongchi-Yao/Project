#%% import libraries
import nltk
import csv
import spacy
from tqdm import tqdm
import torch
import pandas as pd
#%% Load dataset
df=pd.read_json('All_Beauty_5.json', lines=True)
texts = [' '.join([str(i),str(j)]) for i,j in zip(df['reviewText'],df['summary'])]
labels = [i for i in df['overall']]

#%% lemmatize
import myRaw2Lemmatized
lemmatized=myRaw2Lemmatized.Raw2Lemmatized(texts)
#%% Spacy for extracting entity and relation

# get entity pairs like: ['I', 'you']
# input a list of sentences, output a list of entity pairs
# pairs=get_entity(input)
def get_entity(content):
    nlp = spacy.load("en_core_web_sm")
    pairs=[]
    for sent in tqdm(content):
        # chunk 1: Defined a few empty variables in this chunk. prv_tok_dep and prv_tok_text will hold the dependency tag of the previous word in the sentence and that previous word itself, respectively. prefix and modifier will hold the text that is associated with the subject or the object.
        ent1 = ""
        ent2 = ""

        prv_tok_dep = ""  # dependency tag of previous token in the sentence
        prv_tok_text = ""  # previous token in the sentence

        prefix = ""
        modifier = ""

        for tok in nlp(sent):
            # chunk 2: Next, we will loop through the tokens in the sentence. We will first check if the token is a punctuation mark or not. If yes, then we will ignore it and move on to the next token. If the token is a part of a compound word (dependency tag = “compound”), we will keep it in the prefix variable. A compound word is a combination of multiple words linked to form a word with a new meaning (example – “Football Stadium”, “animal lover”).
            # if token is a punctuation mark then move on to the next token
            if tok.dep_ != "punct":
                # check: token is a compound word or not
                if tok.dep_ == "compound":
                    prefix = tok.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        prefix = prv_tok_text + " " + tok.text

                # check: token is a modifier or not
                if tok.dep_.endswith("mod") == True:
                    modifier = tok.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        modifier = prv_tok_text + " " + tok.text

                # chunk 3: Here, if the token is the subject, then it will be captured as the first entity in the ent1 variable. Variables such as prefix, modifier, prv_tok_dep, and prv_tok_text will be reset.
                if tok.dep_.find("subj") == True:
                    ent1 = modifier + " " + prefix + " " + tok.text
                    prefix = ""
                    modifier = ""
                    prv_tok_dep = ""
                    prv_tok_text = ""

                # chunk 4: Here, if the token is the object, then it will be captured as the second entity in the ent2 variable. Variables such as prefix, modifier, prv_tok_dep, and prv_tok_text will again be reset.
                if tok.dep_.find("obj") == True:
                    ent2 = modifier + " " + prefix + " " + tok.text

                # chunk 5: Once we have captured the subject and the object in the sentence, we will update the previous token and its dependency tag.
                # update variables
                prv_tok_dep = tok.dep_
                prv_tok_text = tok.text
        pairs.append([ent1.strip(), ent2.strip()])
    return pairs
pairs=get_entity(texts)

# relation extraction: we need edges
# input a list of sentences, output a list of relation words
# Let’s take a look at the most frequent relations or predicates that we have just extracted: pd.Series(relations).value_counts()[:50]
def get_relation(content):
    nlp = spacy.load("en_core_web_sm")
    relation=[]
    for sent in tqdm(content):
      doc = nlp(sent)

      # Matcher class object
      from spacy.matcher import Matcher
      matcher = Matcher(nlp.vocab)

      #define the pattern
      pattern = [{'DEP':'ROOT'},
                {'DEP':'prep','OP':"?"},
                {'DEP':'agent','OP':"?"},
                {'POS':'ADJ','OP':"?"}]

      matcher.add("matching_1", [pattern])

      matches = matcher(doc)
      k = len(matches) - 1

      span = doc[matches[k][1]:matches[k][2]]
      relation.append(span.text)
    return (relation)
relation=get_relation(texts)

#%% build knowldge graph
source=[i[0] for i in pairs]
target=[i[1] for i in pairs]
kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relation})

# create a directed-graph from the whole dataframe
import networkx as nx
import matplotlib.pyplot as plt
G=nx.from_pandas_edgelist(kg_df, "source", "target",
                           edge_attr=True, create_using=nx.MultiDiGraph())
plt.figure(figsize=(12,12))
pos = nx.spring_layout(G,k=None) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()

# plot one relation
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="break"], "source", "target",
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()

#%% other trivials of spacy library
''
# look up the dependency of the sentence, used in later get_entities function
for tok in doc:
  print(tok.text, "...", tok.dep_)

# look up the entity of the sentence
for ent in doc.ents:
    print(ent.text, ent.label_)
