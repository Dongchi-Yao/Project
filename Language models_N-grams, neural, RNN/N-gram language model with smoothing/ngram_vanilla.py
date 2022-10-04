import numpy as np
from collections import defaultdict


NEG_INFINITY = -20

class NGramVanilla(object):
    def __init__(self, n, vsize):
        self.n = n
        self.count = defaultdict(lambda: defaultdict(float))
        self.total = defaultdict(float)
        self.vsize = vsize
    
    def estimate(self, sequences):
        """Estimate the n-gram counts of order self.n.
        
        Specifically, this function updates self.count and self.total as follows:
          1. self.count[prefix][token] should be the number of occurences of the tuple prefix followed by the
             the string token, across all sentences in sequences. The special tokens "<bos>" can occur in the
             prefix to represent "beginning of sequence". "<eos>" is a special token for "end of sequence".
          2. self.total[prefix] is the total number of occurences of prefix in sequences.
        
        Args:
          sequences: A list of lists, each of which represents a sentence (list of words).
          
        Example:
          Arguments:
            self.n = 2
            sequences = [["hello", "world"], ["hello", "there"]]
          After running:
            self.counts[("hello",)]["world"] == 1
            self.total[("hello",)] == 2
            self.counts[("world",)]["<eos>"] == 1
            self.total[("<bos>",)] == 2
        """

        for i,sequence in enumerate(sequences):
            padded_sequence = ['<bos>']*(self.n-1)+sequence+['<eos>']
            for j in range(len(padded_sequence) - self.n+1):
                ngram=tuple(padded_sequence[j:j+self.n])
                prefix=ngram[:-1]; word=ngram[-1]
                self.total[prefix] +=1 if prefix in self.total else 0
                self.count[prefix][word] += 1 if word in self.count[prefix] else 0


    def ngram_prob(self, ngram):
        """Return the probability of the n-gram estimated by the model."""
        prefix = ngram[:-1]
        word = ngram[-1]
        if self.total[prefix] == 0:
          return 0
        return self.count[prefix][word] / self.total[prefix]

    def sequence_logp(self, sequence):
        padded_sequence = ['<bos>']*(self.n-1) + sequence + ['<eos>']
        total_logp = 0
        for i in range(len(padded_sequence) - self.n+1):
            ngram = tuple(padded_sequence[i:i+self.n])
            logp = np.log2(self.ngram_prob(ngram))
            total_logp += max(NEG_INFINITY, logp)
        return total_logp
