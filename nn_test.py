#!/usr/bin/env python
import argparse
import json
import models
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
import tensorflow as tf
import util
from util import log
from vocab import Vocab

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--args-from-train", default="", 
                    help="output line from STATS of train that sets (and overrides) opts to repro a training run." +
                         " just need to set test-set & num-from-test. probably want to set --disable-gpu also.")
parser.add_argument("--test-set", default="data/snli_1.0_test.jsonl")
parser.add_argument("--num-from-test", default=-1, type=int,
                    help='number of batches to read from test. -1 => all')
parser.add_argument('--batch-size', default=-1, type=int, help='batch size')
parser.add_argument("--restore-ckpt", default="", help="if set, restore from this ckpt file")
parser.add_argument('--disable-gpu', action='store_true', help='if set we only run on cpu')
parser.add_argument('--vocab-file', 
                    help='vocab (token -> idx) for embeddings,'
                         ' required if using --initial-embeddings')

model = models.BidirGruConcatMlp()
model.add_args(parser)
opts = parser.parse_args()
if not opts.dont_train_embeddings:
    log("overriding --dont-train-embeddings to True")
    opts.dont_train_embeddings = True
if opts.args_from_train:
    args = json.loads(opts.args_from_train)
    opts.batch_size = args['batch_size']
    opts.embedding_dim = args['embedding_dim']
    opts.hidden_dim = args['hidden_dim']
    opts.initial_embeddings = args['initial_embeddings']
    opts.mlp_config = args['mlp_config']
    opts.seq_len = args['seq_len']
    opts.restore_ckpt = args['last_ckpt']
    opts.vocab_file = args['output_vocab_file']
print >>sys.stderr, opts

log("loading data")
vocab = Vocab(opts.vocab_file)
test_x, test_y, test_stats = util.load_data(opts.test_set, vocab,
                                         update_vocab=False,
                                         max_records=opts.num_from_test,
                                         max_len=opts.seq_len,
                                         batch_size=opts.batch_size)
log("test_stats %s %s" % (len(test_x), test_stats))
log("|VOCAB| %s" % vocab.size())

log("building model")
# construct 4 rnns; one for each of a forwards/backwards pass over s1/s2
# TODO: while they are all the same length could pack them into single tensor
# TODO: explicitly did sX_f/b seperately for ease of data gen, could construct these in tf graph & just have s1, s2
s1_f, s1_b, s2_f, s2_b = model.input_placeholders(opts)
logits, predicted = model.logits(vocab, opts)

def eg_to_feeddict(eg):
    return {s1_f: eg['s1_f'], s1_b: eg['s1_b'],
            s2_f: eg['s2_f'], s2_b: eg['s2_b']}

log("creating session")
sess = tf.Session()
c = tf.ConfigProto()
if opts.disable_gpu:
    c.device_count['GPU'] = 0
sess = tf.Session(config=c)
sess.run(tf.initialize_all_variables())

log("restoring from ckpt %s" % opts.restore_ckpt)
saver = tf.train.Saver()
saver.restore(sess, opts.restore_ckpt)

predicteds, actuals = [], []
for eg, true_labels in zip(test_x, test_y):
    b_predicted = sess.run(predicted, feed_dict=eg_to_feeddict(eg))
    for p, a in zip(b_predicted, true_labels):
        predicteds.append(np.argmax(p))
        actuals.append(np.argmax(a))
dev_c = confusion_matrix(actuals, predicteds)
dev_accuracy = util.accuracy(dev_c)
print "dev confusion\n %s (%s)" % (dev_c, dev_accuracy)


