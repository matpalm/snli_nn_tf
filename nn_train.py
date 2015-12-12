#!/usr/bin/env python
import argparse
import json
import models
import numpy as np
import os
import random
from sklearn.metrics import confusion_matrix
from stats import Stats
import sys
import tensorflow as tf
from tensorflow.models.rnn import linear
import time
import util
from util import log
from vocab import Vocab

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--train-set", default="data/snli_1.0_train.jsonl")
parser.add_argument("--num-from-train", default=-1, type=int,
                    help='number of batches to read from train. -1 => all')
parser.add_argument("--dev-set", default="data/snli_1.0_dev.jsonl")
parser.add_argument("--num-from-dev", default=-1, type=int,
                    help='number of batches to read from dev. -1 => all')
parser.add_argument("--dev-run-freq", default=10000, type=int,
                    help='frequency (in num batches trained) to run against dev set')
parser.add_argument('--batch-size', default=10, type=int,
                    help='batch size')
parser.add_argument("--num-epochs", default=-1, type=int,
                    help='number of epoches to run. -1 => forever')
parser.add_argument("--optimizer", default="GradientDescentOptimizer",
                    help='optimizer to use; some baseclass of tf.train.Optimizer')
parser.add_argument("--learning-rate", default=0.01, type=float)
parser.add_argument("--momentum", default=0, type=float, help="momentum (for MomentumOptimizer)")
parser.add_argument("--restore-ckpt", default="", help="if set, restore from this ckpt file")
parser.add_argument("--ckpt-dir", default="", help="root dir to save ckpts. blank => don't save ckpts")
parser.add_argument("--ckpt-freq", default=100000, type=int,
                    help='frequency (in num batches trained) to dump ckpt to --ckpt-dir')
parser.add_argument('--disable-gpu', action='store_true', help='if set we only run on cpu')
parser.add_argument('--input-vocab-file',
                    help='vocab (token -> idx) for embeddings,'
                         ' required if using --initial-embeddings')
parser.add_argument('--output-vocab-file',
                    help='derived vocab as updated from loading training data. used for nn_test')

model = models.BidirGruConcatMlp()
model.add_args(parser)
opts = parser.parse_args()
print >>sys.stderr, opts

# check that if one of --vocab--file or --initial_embeddings is set, they are both set.
assert not ((opts.input_vocab_file is None) ^ (opts.initial_embeddings is None)), "must set both --vocab-file & --initial-embeddings"

# build vocab and load data
log("loading data")
vocab = Vocab(opts.input_vocab_file)
train_x, train_y, train_stats = util.load_data(opts.train_set, vocab,
                                               update_vocab=True,
                                               max_records=opts.num_from_train,
                                               max_len=opts.seq_len,
                                               batch_size=opts.batch_size)
log("train_stats %s %s" % (len(train_x), train_stats))
dev_x, dev_y, dev_stats = util.load_data(opts.dev_set, vocab,
                                         update_vocab=False,
                                         max_records=opts.num_from_dev,
                                         max_len=opts.seq_len,
                                         batch_size=opts.batch_size)
log("dev_stats %s %s" % (len(dev_x), dev_stats))
log("|VOCAB| %s" % vocab.size())
if opts.output_vocab_file:
    vocab.write_to(opts.output_vocab_file)

log("building model")
# construct 4 rnns; one for each of a forwards/backwards pass over s1/s2
# TODO: while they are all the same length could pack them into single tensor
# TODO: explicitly did sX_f/b seperately for ease of data gen, could construct these in tf graph & just have s1, s2
s1_f, s1_b, s2_f, s2_b = model.input_placeholders(opts)
logits, predicted = model.logits(vocab, opts)

labels = tf.placeholder(tf.float32, [opts.batch_size, 3])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
mean_cross_entropy = tf.reduce_mean(cross_entropy)
cost = mean_cross_entropy  # TODO: add reg
optimizer = util.optimizer(opts).minimize(cost)

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
c = tf.ConfigProto()
if opts.disable_gpu:
    c.device_count['GPU'] = 0
sess = tf.Session(config=c)
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
    full_ckpt_path = os.path.join(opts.ckpt_dir, RUN_ID, ckpt_file)
    stats.last_ckpt = full_ckpt_path
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
        # occasionally write a checkpoint
        if opts.ckpt_dir and stats.n_batches_trained % opts.ckpt_freq == 0:
            save_ckpt()
        # occasionally check dev set
        if stats.n_batches_trained % opts.dev_run_freq == 0:
            stats_from_dev_set(stats)
            stats.flush_to_stdout(epoch)
    epoch += 1

# final flush of stats and final ckpt save
stats_from_dev_set(stats)
stats.flush_to_stdout(epoch)
if opts.ckpt_dir:
    save_ckpt()
