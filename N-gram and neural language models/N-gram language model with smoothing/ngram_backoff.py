import numpy as np

from ngram_vanilla import NGramVanilla


class NGramBackoff(NGramVanilla):
    def __init__(self, n, vsize):
        self.n = n
        self.sub_models = [NGramVanilla(k, vsize) for k in range(1, n + 1)]

    def estimate(self, sequences):
        for sub_model in self.sub_models:
            sub_model.estimate(sequences)

    def ngram_prob(self, ngram):
        """Return the smoothed probability with backoff.
        
        That is, if the n-gram count of size self.n is defined, return that.
        Otherwise, check the n-gram of size self.n - 1, self.n - 2, etc. until you find one that is defined.
        
        Hint: Refer to ngram_prob in ngrams_vanilla.py.
        """

        for i in range(self.n-1,-1,-1):
            prob=self.sub_models[i].ngram_prob(ngram[-i-1:])
            if prob!=0: break
        return prob
