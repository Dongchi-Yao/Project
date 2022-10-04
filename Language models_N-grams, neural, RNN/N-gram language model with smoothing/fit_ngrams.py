import os
import json
import argparse
import requests

from ngram_vanilla import NGramVanilla
from ngram_additive import NGramAdditive
from ngram_interpolation import NGramInterpolation
from ngram_backoff import NGramBackoff



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("smoothing", choices=["vanilla", "additive", "interpolation", "backoff"])
    parser.add_argument("--n", "-n", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.0005)
    parser.add_argument("--lambda1", type=float, default=1/3)
    parser.add_argument("--lambda2", type=float, default=1/3)
    parser.add_argument("--lambda3", type=float, default=1/3)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def load_wikitext(filename='wikitext2-sentencized.json'):
    if not os.path.exists(filename):
        url = 'https://nyu.box.com/shared/static/9kb7l7ci30hb6uahhbssjlq0kctr5ii4.json'
        r = requests.get(url)
        with open(filename, "wb") as fh:
            fh.write(r.content)
    
    datasets = json.load(open(filename, 'r'))
    for name in datasets:
        datasets[name] = [x.split() for x in datasets[name]]
    vocab = list(set([t for ts in datasets['train'] for t in ts]))      
    print("Vocab size: %d" % (len(vocab)))
    return datasets, vocab


def perplexity(model, sequences):
    n_total = 0
    logp_total = 0
    for sequence in sequences:
        logp_total += model.sequence_logp(sequence)
        n_total += len(sequence) + 1  
    ppl = 2 ** (- (1.0 / n_total) * logp_total)  
    return ppl


def get_lm(args):
    if args.smoothing == "vanilla":
        return NGramVanilla(n=args.n, vsize=len(vocab)+1)
    elif args.smoothing == "additive":
        return NGramAdditive(n=args.n, delta=args.delta, vsize=len(vocab)+1)  # +1 is for <eos>
    elif args.smoothing == "interpolation":
        lambdas = [args.lambda1, args.lambda2, args.lambda3]
        return NGramInterpolation(lambdas=lambdas, vsize=len(vocab)+1)  # +1 is for <eos>
    elif args.smoothing == "backoff":
        return NGramBackoff(n=args.n, vsize=len(vocab) + 1)
    else:
        raise NotImplementedError("Unknown smoothing method: " + args.smoothing)


if __name__ == "__main__":
    args = parse_args()
    datasets, vocab = load_wikitext()
    lm = get_lm(args)
    lm.estimate(datasets['train'])
    if args.test:
        print("Test Perplexity: %.3f" % perplexity(lm, datasets['test']))
    else:
        print("Train Perplexity: %.3f" % perplexity(lm, datasets['train']))
        print("Valid Perplexity: %.3f" % perplexity(lm, datasets['valid']))
