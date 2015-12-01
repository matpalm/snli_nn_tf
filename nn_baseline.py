#!/usr/bin/env python
import argparse
import json
import numpy as np
import random
import sys
import tensorflow as tf
from tensorflow.models.rnn import linear
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
import util
from vocab import Vocab

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--train-set", default="data/snli_1.0_train.jsonl")
parser.add_argument("--num-from-train", default=-1, type=int,
                    help='number of batches to read from train. -1 => all')
parser.add_argument('--batch-size', default=10, type=int,
                    help='batch size')
parser.add_argument("--hack-max-len", default=4,
                    help="hack; need to do bucketing, for now just ignore long egs")
parser.add_argument('--hidden-dim', default=100, type=int,
                    help='hidden node dimensionality')
parser.add_argument('--embedding-dim', default=10, type=int,
                    help='embedding node dimensionality')
parser.add_argument("--num-epochs", default=-1, type=int,
                    help='number of epoches to run. -1 => forever')
opts = parser.parse_args()
print >>sys.stderr, opts
num_from_train = int(opts.num_from_train)
seq_len = int(opts.hack_max_len)
hidden_dim = int(opts.hidden_dim)
embedding_dim = int(opts.embedding_dim)
batch_size = int(opts.batch_size)

def log(s):
    print >>sys.stderr, util.dts(), s

# build vocab and load data
log("loading data")
vocab = Vocab() #opts.vocab_file)
train_x, train_y, train_stats = util.load_data(opts.train_set,
                                               max_records=num_from_train,
                                               max_len=seq_len,
                                               batch_size=batch_size,
                                               vocab=vocab)
print "|VOCAB|", vocab.size()

print "building model"
# construct 4 rnns; one for each of a forwards/backwards pass over s1/s2
s1_f = tf.placeholder(tf.int32, [batch_size, seq_len])
s1_b = tf.placeholder(tf.int32, [batch_size, seq_len])
s2_f = tf.placeholder(tf.int32, [batch_size, seq_len])
s2_b = tf.placeholder(tf.int32, [batch_size, seq_len])
inputs = [s1_f, s1_b, s2_f, s2_b]

# embed each of the s1/s2 f/b passes
# for now using the same embedding matrix
with tf.device("/cpu:0"):
    embeddings = tf.get_variable("embeddings", [vocab.size(), embedding_dim])
def embedded_sequence(seq):
    with tf.device("/cpu:0"):
        # embed entire sequence in one hit
        embedded_inputs = tf.nn.embedding_lookup(embeddings, seq)  # (batch_size, seq_len, embedding_dim)
        # unpack on seq_len dimension (1) into an array of len seq_len
        inputs = tf.split(1, seq_len, embedded_inputs)  # [(batch_size, 1, embedding_dim), ...]
        # squeeze each element (ie throw away "empty dimension"). collapses from 3d to 2d
        inputs = [tf.squeeze(i) for i in inputs]  # [(batch_size, embedding_dim), ...]
        return inputs
embedded_inputs = [embedded_sequence(s) for s in inputs]

# build an rnn over each sequence capturing the final state from each.
def final_state_of_rnn_over_embedded_sequence(idx, embedded_seq):
    with tf.variable_scope("rnn_%s" % idx):
        gru = rnn_cell.GRUCell(hidden_dim)
        initial_state = gru.zero_state(batch_size, tf.float32)
        outputs, _states = rnn.rnn(gru, embedded_seq, initial_state=initial_state)
        return outputs[-1]
final_states = [final_state_of_rnn_over_embedded_sequence(idx, s) for idx, s in enumerate(embedded_inputs)]

# concat these states into a single matrix
# i.e. go from len=4 [(batch_size, hidden), ...] to single element
concatted_final_states = tf.concat(1, final_states)  # (batch_size, hidden_dim*4)

# run these through a logisitic regression
with tf.variable_scope("lr"):
    projection = tf.get_variable("projection", [hidden_dim*4, 3])  # 3 labels
    bias = tf.get_variable("bias", [3], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(concatted_final_states, projection) + bias

# calculate cost and assign trainer
labels = tf.placeholder(tf.float32, [batch_size, 3])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
mean_cross_entropy = tf.reduce_mean(cross_entropy)
cost = mean_cross_entropy  # TODO: add reg
sgd = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

# also provide a hook for softmax output
predicted = tf.nn.softmax(logits)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

log("training")
epoch = 0
egs = zip(train_x, train_y)
while epoch != int(opts.num_epochs):
    print "EPOCH", epoch
    random.shuffle(egs)
    for n, (eg, true_labels) in enumerate(egs):
        # TODO: move this into ONE matrix and slice these things out.
        # TODO: make use of hidden->hidden masks
        fd = {s1_f: eg['s1_f'], s1_b: eg['s1_b'],
              s2_f: eg['s2_f'], s2_b: eg['s2_b'],
              labels: true_labels}
        sess.run(sgd, feed_dict=fd)

        if (n%5) == 0:
            print "batch #", n
            print "true", true_labels
            pred = sess.run(predicted, feed_dict=fd)
            print "predicted", pred
            print "cost (pre training)", sess.run(cost, feed_dict=fd)

    epoch += 1

