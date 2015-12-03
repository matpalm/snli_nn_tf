#!/usr/bin/env python
import argparse
import json
import numpy as np
import os
import random
from sklearn.metrics import confusion_matrix
from stats import Stats
import sys
import tensorflow as tf
from tensorflow.models.rnn import linear
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
import time
import util
from vocab import Vocab

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--train-set", default="data/snli_1.0_train.jsonl")
parser.add_argument("--num-from-train", default=-1, type=int,
                    help='number of batches to read from train. -1 => all')
parser.add_argument("--dev-set", default="data/snli_1.0_dev.jsonl")
parser.add_argument("--num-from-dev", default=-1, type=int,
                    help='number of batches to read from dev. -1 => all')
parser.add_argument("--dev-run-freq", default=100000, type=int,
                    help='frequency (in num batches trained) to run against dev set')
parser.add_argument('--batch-size', default=10, type=int,
                    help='batch size')
parser.add_argument("--hack-max-len", default=25, type=int,
                    help="hack; need to do bucketing, for now just ignore long egs")
parser.add_argument('--hidden-dim', default=100, type=int,
                    help='hidden node dimensionality')
parser.add_argument('--embedding-dim', default=100, type=int,
                    help='embedding node dimensionality')
parser.add_argument("--num-epochs", default=-1, type=int,
                    help='number of epoches to run. -1 => forever')
parser.add_argument("--optimizer", default="GradientDescentOptimizer",
                    help='optimizer to use; some baseclass of tf.train.Optimizer')
parser.add_argument("--learning-rate", default=0.01, type=float)
parser.add_argument("--momentum", default=0, type=float, help="momentum (for MomentumOptimizer)")
parser.add_argument("--mlp-config", default="[35]",
                    help="pre classifier mlp config; array describing #hidden nodes"
                         + " per layer. eg [50,50,20] denotes 3 hidden layers, with 50, 50 and 20"
                         + " nodes. a value of [] denotes no MLP before classifier")
parser.add_argument("--restore-ckpt", default="", help="if set, restore from this ckpt file")
parser.add_argument("--ckpt-dir", default="", help="root dir to save ckpts. blank => don't save ckpts")
parser.add_argument("--ckpt-freq", default=100000, type=int,
                    help='frequency (in num batches trained) to dump ckpt to --ckpt-dir')

opts = parser.parse_args()
print >>sys.stderr, opts
seq_len = int(opts.hack_max_len)
hidden_dim = int(opts.hidden_dim)
embedding_dim = int(opts.embedding_dim)
batch_size = int(opts.batch_size)

def log(s):
    print >>sys.stderr, util.dts(), s

# build vocab and load data
log("loading data")
vocab = Vocab() #opts.vocab_file)
train_x, train_y, train_stats = util.load_data(opts.train_set, vocab,
                                               update_vocab=True,
                                               max_records=opts.num_from_train,
                                               max_len=seq_len,
                                               batch_size=batch_size)
log("train_stats %s %s" % (len(train_x), train_stats))
dev_x, dev_y, dev_stats = util.load_data(opts.dev_set, vocab,
                                         update_vocab=False,
                                         max_records=opts.num_from_dev,
                                         max_len=seq_len,
                                         batch_size=batch_size)
log("dev_stats %s %s" % (len(dev_x), dev_stats))
log("|VOCAB| %s" % vocab.size())

log("building model")
# construct 4 rnns; one for each of a forwards/backwards pass over s1/s2
# TODO: while they are all the same length could pack them into single tensor
# TODO: explicitly did sX_f/b seperately for ease of data gen, could construct these in tf graph & just have s1, s2
s1_f = tf.placeholder(tf.int32, [batch_size, seq_len])  # forward over s1
s1_b = tf.placeholder(tf.int32, [batch_size, seq_len])  # backwards over s1
s2_f = tf.placeholder(tf.int32, [batch_size, seq_len])
s2_b = tf.placeholder(tf.int32, [batch_size, seq_len])
inputs = [s1_f, s1_b, s2_f, s2_b]

# embed each of the s1/s2 f/b passes
# for now using the same embedding matrix
with tf.device("/cpu:0"):
    vs, ed = vocab.size(), embedding_dim
    embeddings = tf.get_variable("embeddings", [vs, ed])
    # zero out the 0th entry (PAD_ID) (we do this with the assumption
    # that network will emit 0 hidden state for zero hidden t-1 state
    # & zero embedding.)
    # TODO this is a clumsy way to do it :/
    embeddings *= np.asarray([0]*ed + [1]*((vs-1)*ed)).reshape(vs, ed)

def embedded_sequence(seq):
    with tf.device("/cpu:0"):
        # embed entire sequence in one hit
        # TODO: do all 4 sequences in one hit.
        # TODO: sX_f and sX_b embed the same sequences, just do them once
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

# (optionally) add some hidden layers between concatted_final_states and output.
# note: we always have a final logisitic regression to 3 classes.
# eg opts.mlp_config = [200, 200] => two hidden layers; concatted states -> 200d -> 200d -> 3d
# eg opts.mlp_config = [] => no hidden layers; just concatted states -> 3d
def mlp_layer(name, input, input_dim, output_dim, include_nonlinearity=True):
    with tf.variable_scope(name):
        projection = tf.get_variable("projection", [input_dim, output_dim])
        bias = tf.get_variable("bias", [1, output_dim], initializer=tf.constant_initializer(0.0))
        output = tf.matmul(input, projection) + bias  # note: bias is broadcast across leading batch dim
        return tf.nn.relu(output) if include_nonlinearity else output
last_layer = concatted_final_states
last_layer_size = hidden_dim * 4
for n, num_hidden_nodes in enumerate(eval(opts.mlp_config)):
    last_layer = mlp_layer("mlp_hidden_%d" % n, last_layer, last_layer_size, num_hidden_nodes)
    last_layer_size = num_hidden_nodes
logits = mlp_layer("logits", last_layer, last_layer_size, 3, include_nonlinearity=False)

# calculate cost and assign trainer
labels = tf.placeholder(tf.float32, [batch_size, 3])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
mean_cross_entropy = tf.reduce_mean(cross_entropy)
cost = mean_cross_entropy  # TODO: add reg
optimizer = util.optimizer(opts).minimize(cost)

# also provide a hook for softmax output
predicted = tf.nn.softmax(logits)

def eg_and_label_to_feeddict(eg, true_labels):
    return {s1_f: eg['s1_f'], s1_b: eg['s1_b'],
            s2_f: eg['s2_f'], s2_b: eg['s2_b'],
            labels: true_labels}

def stats_from_dev_set(stats):
    predicteds, actuals = [], []
    for eg, true_labels in zip(dev_x, dev_y):
        b_cost, b_predicted = sess.run([cost, predicted],
                                       feed_dict=eg_and_label_to_feeddict(eg, true_labels))
        stats.record_dev_cost(b_cost)
        for p, a in zip(b_predicted, true_labels):
            predicteds.append(np.argmax(p))
            actuals.append(np.argmax(a))
    dev_c = confusion_matrix(actuals, predicteds)
    dev_accuracy = util.accuracy(dev_c)
    stats.set_dev_accuracy(dev_accuracy)
    print "dev confusion\n %s (%s)" % (dev_c, dev_accuracy)


RUN_ID = "RUN_%s_%s" % (int(time.time()), os.getpid())

log("creating session")
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# setup saver
saver = None
if opts.restore_ckpt or opts.ckpt_dir:
    saver = tf.train.Saver()
if opts.restore_ckpt:
    log("restoring from ckpt %s" % opts.restore_ckpt)
    saver.restore(sess, opts.restore_ckpt)
if opts.ckpt_dir:
    os.mkdir(os.path.join(opts.ckpt_dir, RUN_ID))

stats = Stats(os.path.basename(__file__), opts, RUN_ID)

def save_ckpt():
    ckpt_file = str(int(time.time()))
    stats.last_ckpt = ckpt_file
    full_ckpt_path = os.path.join(opts.ckpt_dir, RUN_ID, ckpt_file)
    saver.save(sess, full_ckpt_path)
    log("ckpt saved to %s" % full_ckpt_path)

log("training")
epoch = 0
egs = zip(train_x, train_y)
last_ckpt = ""
while epoch != int(opts.num_epochs):
    random.shuffle(egs)
    for n, (eg, true_labels) in enumerate(egs):
        # train a batch
        # TODO: move this into ONE matrix and slice these things out.
        # TODO: move away from feeddict and to queueing egs.
        batch_cost, _opt = sess.run([cost, optimizer],
                                    feed_dict=eg_and_label_to_feeddict(eg, true_labels))
        stats.record_training_cost(batch_cost)
        # occasionally check dev set
        if stats.n_batches_trained % opts.dev_run_freq == 0:
            stats_from_dev_set(stats)
            stats.flush_to_stdout(epoch)
        # occasionally write a checkpoint
        if opts.ckpt_dir and stats.n_batches_trained % opts.ckpt_freq == 0:
            save_ckpt()
    epoch += 1

if opts.ckpt_dir:
    save_ckpt()
