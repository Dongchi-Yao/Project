# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralNet(nn.Module):

    def __init__(self, embeddings: nn.Embedding, hidden_dim: int):
        super().__init__()
        self.embeddings = embeddings
        self.hidden_dim = hidden_dim
        # TODO: Your code here!
        self.hidden_layer=nn.Linear(self.embeddings.embedding_dim, self.hidden_dim)
        self.classifier=nn.Linear(self.hidden_dim,2)
    
    def forward(self, ex_words: List[str]) -> torch.Tensor:
        """Takes a list of words and returns a tensor of class predictions.

        Steps:
          1. Embed each word using self.embeddings.
          2. Take the average embedding of each word.
          3. Pass the average embedding through a neural net with one hidden layer (with ReLU nonlinearity).
          4. Return a tensor representing a log-probability distribution over classes.
        """
        embed=self.embeddings(torch.tensor(ex_words)) # [seq,embed]
        average_embed=torch.mean(embed,dim=-2) #[embed]
        logits=self.hidden_layer(average_embed) #[hidden]
        logits=nn.functional.relu(logits) #[hidden]
        logits=self.classifier(logits) #[2]
        return torch.nn.functional.log_softmax(logits, dim=-1) #[2]


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, hidden_dim: int, word_embeddings):
        self.hidden_dim = hidden_dim
        # TODO: Your code here
        self.word_embeddings=word_embeddings
        self.model = NeuralNet(self.word_embeddings.get_initialized_embedding_layer(), self.hidden_dim)
        self.logits = None

    def predict(self, ex_words: List[str]) -> int:
        # TODO: Your code here!
        list_indices=[]
        for word in ex_words:
            tmp=self.word_embeddings.word_indexer.index_of(word)
            index=tmp if tmp!=-1 else self.word_embeddings.word_indexer.index_of('UNK')
            list_indices.append(index)
        self.logits=self.model(list_indices)
        return torch.argmax(self.logits,-1)


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # TODO: Your code here!
    torch.manual_seed(0)
    model=NeuralSentimentClassifier(args.hidden_size,word_embeddings)
    criterion=nn.NLLLoss()
    torch.manual_seed(0)
    optimizer = optim.SGD(model.model.parameters(), lr=args.lr)
    import random
    random.Random(0).shuffle(train_exs)
    for j in range(args.num_epochs):
        train_loss_list = 0 # just for the purpose of tracking training loss
        for i in train_exs:
            optimizer.zero_grad()
            model.model.train()
            y_hat_label=model.predict(i.words)
            y_hat=model.logits
            train_loss = criterion(y_hat, torch.tensor(i.label).long())
            train_loss.backward()
            train_loss_list+=train_loss.item() # just for the purpose of tracking training loss
            optimizer.step()
        
        # the following codes are just for the purpose of tracking the training loss during my development, along with several codes above. 
        '''
        N=0
        for i in train_exs:
            model.model.eval()
            y_hat_label = model.predict(i.words)
            y_label=i.label
            if y_hat_label==y_label: N+=1

        print('epoch', j + 1)
        print('train_loss', train_loss_list / len(train_exs))
        print ('train_acc', N/len(train_exs))
        '''
    return model

