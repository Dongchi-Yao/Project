# Homework 1 Instructions

NLP with Representation Learning, Fall 2022

Name: Daniel Yao

Net ID: dy2037

## Part 1: N-Gram Model

### Part 1a

Vocab size: 33175
Train Perplexity: 8.107
C:\Users\smrya\PycharmProjects\pythonProject\NLP HW\hw1-9-19\hw1\ngram_vanilla.py:59: RuntimeWarning: divide by zero encountered in log2
  logp = np.log2(self.ngram_prob(ngram))
Valid Perplexity: 10121.316

## Part 2: N-Gram Model with Additive Smoothing

### Part 2a

Answer: When we are computing the N-gram probability P(w|previous n-1 words), it is very likely that the previous n-1 words do not appear in the training set. In this case, if we use Bayesian equation and Maximum Likelihood Estimation, we would treat the count of (w, previous n-1) words as 0, and of course it is not reasonable. Therefore, we consider to add a small value to these zero counts to avoid the sparsity.  

### Part 2b

Vocab size: 33175
Train Perplexity: 116.398
Valid Perplexity: 2844.227

### PART 2c

(2,0.05): 663.045
(2,0.005): 447.895
(2,0.0005): 440.781
(2,0.0004): 446.065
(2,0.00005):508.939
(3,0.05): 4648.274
(3,0.005): 2844.227
(3,0.0005): 2421.402
(3,0.0004): 2434.057
(3,0.00005): 2981.892

From the results above, I would choose n=2, delta=0.0005, since it has the lowest validation perplexity among all of my trials. 



## Part 3: N-Gram Model with Interpolation Smoothing

### Part 3a

Vocab size: 33175
Train Perplexity: 17.596
Valid Perplexity: 293.566

### Part 3b

**TODO**: Report validation perplexity for different lambda values and select best lambdas.
lambda1=0.2, lambda2=0.5, lambda3=0.3 ----validation perplexity: 288.764
lambda1=0.2, lambda2=0.6, lambda3=0.2 ----validation perplexity: 282.855
lambda1=0.2, lambda2=0.65, lambda3=0.15 ----validation perplexity: 282.158 *[the best]
lambda1=0.2, lambda2=0.7, lambda3=0.1 ----validation perplexity: 283.434
lambda1=0.25, lambda2=0.4, lambda3=0.35 ----validation perplexity: 292.189

From the results above, the best combination of the lambdas are [0.2, 0.65, 0.15], as its validation perplexity is the lowest among all of my trials.


## Part 4: Backoff

### Part 4a

Vocab size: 33175
Train Perplexity: 8.107
Valid Perplexity: 142.995

## Part 5: Test Set Evaluation

### Part 5a

**TODO**: Test set perplexity for each model type. Indicate best model.

Vanilla model:
n=1, 919.534 
n=2, 513.943 (best model)
n=3, 9279.365

Additive smoothing (the best model selected from above, n=2, delta=0.0005):
412.149

Interpolation smoothing (the best model selected from above, lambdas=[0.2, 0.65, 0.15])
266.805

Backoff smoothing:
136.922

From the results above, models with backoff smoothing is the best, because it has the lowest perplexity score 136.922 on test set. 


### Part 6: A Taste of Neural Networks

Notes: with default values of argument parameters
=====Train Accuracy=====
Accuracy: 5014 / 6920 = 0.724566;
Precision (fraction of predicted positives that are correct): 2582 / 3460 = 0.746243;
Recall (fraction of true positives predicted correctly): 2582 / 3610 = 0.715235;
F1 (harmonic mean of precision and recall): 0.730410;

=====Dev Accuracy=====
Accuracy: 625 / 872 = 0.716743;
Precision (fraction of predicted positives that are correct): 334 / 471 = 0.709130;
Recall (fraction of true positives predicted correctly): 334 / 444 = 0.752252;
F1 (harmonic mean of precision and recall): 0.730055;
