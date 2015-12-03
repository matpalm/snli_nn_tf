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

### bidir grus -> concat -> mlp -> logreg

first model is 
* a bidirictional gru rnn over each sentence
* concatted final states -> couple layer mlp -> 3way logisitic regression

```
$ ./nn_baseline.py --help
usage: nn_baseline.py [-h] [--train-set TRAIN_SET]
                      [--num-from-train NUM_FROM_TRAIN] [--dev-set DEV_SET]
                      [--num-from-dev NUM_FROM_DEV]
                      [--dev-run-freq DEV_RUN_FREQ] [--batch-size BATCH_SIZE]
                      [--hack-max-len HACK_MAX_LEN] [--hidden-dim HIDDEN_DIM]
                      [--embedding-dim EMBEDDING_DIM]
                      [--num-epochs NUM_EPOCHS] [--optimizer OPTIMIZER]
                      [--learning-rate LEARNING_RATE] [--momentum MOMENTUM]
                      [--mlp-config MLP_CONFIG] [--restore-ckpt RESTORE_CKPT]
                      [--ckpt-dir CKPT_DIR] [--ckpt-freq CKPT_FREQ]

optional arguments:
  -h, --help                       show this help message and exit
  --train-set TRAIN_SET
  --num-from-train NUM_FROM_TRAIN  number of batches to read from train. -1 => all
  --dev-set DEV_SET
  --num-from-dev NUM_FROM_DEV      number of batches to read from dev. -1 => all
  --dev-run-freq DEV_RUN_FREQ      frequency (in num batches trained) to run against dev set
  --batch-size BATCH_SIZE          batch size
  --hack-max-len HACK_MAX_LEN      hack; need to do bucketing, for now just ignore long egs
  --hidden-dim HIDDEN_DIM          hidden node dimensionality
  --embedding-dim EMBEDDING_DIM    embedding node dimensionality
  --num-epochs NUM_EPOCHS          number of epoches to run. -1 => forever
  --optimizer OPTIMIZER            optimizer to use; some baseclass of tr.train.Optimizer
  --learning-rate LEARNING_RATE
  --momentum MOMENTUM              momentum (for MomentumOptimizer)
  --mlp-config MLP_CONFIG          pre classifier mlp config; array describing #hidden
                                   nodes per layer. eg [50,50,20] denotes 3 hidden
                                   layers, with 50, 50 and 20 nodes. a value of []
                                   denotes no MLP before classifier
  --restore-ckpt RESTORE_CKPT      if set, restore from this ckpt file
  --ckpt-dir CKPT_DIR              root dir to save ckpts. blank => don't save ckpts
  --ckpt-freq CKPT_FREQ            frequency (in num batches trained) to dump ckpt to --ckpt-dir
```
