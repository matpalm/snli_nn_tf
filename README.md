# snli hacking (in tensorflow)

note: port of [snli hacking in theano](https://github.com/matpalm/snli_nn)

hacking with the [Stanford Natural Language Inference corpus](http://nlp.stanford.edu/projects/snli/). problem is to decide if two sentences are neutral, contradict each other or the first entails the second.

## baseline

simple logistic regression using token features, tokens from sentence 1 prepended with
"s1_", tokens from sentence 2 prepended with "s2_"

```
$ time ./log_reg_baseline.py

train confusion
 [[121306  27808  34073]
  [ 29941 117735  35088]
  [ 23662  20907 138847]] (accuracy 0.687)

dev confusion
 [[2077  549  652]
  [ 546 2044  645]
  [ 474  404 2451]] (accuracy 0.667)

# approx 6m
```

## nn models

lots more variants explored in 
[the theano version of this project](https://github.com/matpalm/snli_nn).

### unidir

first model is 
* a unidir gru rnn over each sentence
* concatted to a single logistic regression to a 3way softmax

```
usage: ./nn_baseline.py --num-from-train=10 --num-epochs=1 --embedding-dim=5 --hidden-dim=5 --batch-size=3
```
