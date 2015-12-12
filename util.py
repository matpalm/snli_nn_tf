from collections import Counter
import json
import numpy as np
import sys
import time
import tensorflow as tf

def split_binary_parse(parse, include_parenthesis=False):
    tokens = []
    for token in parse.split(" "):
        if token == "(" or token == ")":
            if include_parenthesis:
                tokens.append(token)
        else:
            tokens.append(token.lower())
    return tokens

def binary_parses(d):
    s1 = split_binary_parse(d['sentence1_binary_parse'])
    s2 = split_binary_parse(d['sentence2_binary_parse'])
    return s1, s2

def padded_forwards_backwards(ids, pad_length, pad_id):
    padding = [pad_id] * (pad_length - len(ids))
    f = padding + ids
    b = padding + list(reversed(ids))
    return f, b

LABELS = ['contradiction', 'neutral', 'entailment']

def label_for(eg):  # TODO: uber clumstastic :/
    try:
        idx = LABELS.index(eg['gold_label'])
        one_hot = [0,0,0]
        one_hot[idx] = 1
        return one_hot
    except ValueError:
        return None

def log(s):
    print >>sys.stderr, dts(), s

def load_data(file, vocab, update_vocab, max_records, max_len, batch_size):
    stats = Counter()

    # final batched data.
    x, y = [], []

    # batch (in progress)
    s1_fs, s1_bs = [], []
    s2_fs, s2_bs = [], []
    labels = []

    for line in open(file, 'r').readlines():
        d = json.loads(line)

        label = label_for(d)
        if label is None:
            stats['bad_label'] += 1
            continue

        s1, s2 = binary_parses(d)
        if len(s1) > max_len or len(s2) > max_len:
            stats['n_ignore_long'] += 1
            continue

        s1_ids = vocab.ids_for_tokens(s1, update_vocab)
        s2_ids = vocab.ids_for_tokens(s2, update_vocab)

        # TODO: urgh. this is horrible; dup data and things that could be done on gpu (padding, reversal, single feed_dict input)

        s1_f, s1_b = padded_forwards_backwards(s1_ids, pad_length=max_len, pad_id=vocab.PAD_ID)
        s2_f, s2_b = padded_forwards_backwards(s2_ids, pad_length=max_len, pad_id=vocab.PAD_ID)

        s1_fs.append(s1_f)
        s1_bs.append(s1_b)
        s2_fs.append(s2_f)
        s2_bs.append(s2_b)
        labels.append(label)

        if len(s1_fs) == batch_size:
            # flush batch
            x.append({"s1_f": np.asarray(s1_fs).reshape(batch_size, max_len),
                      "s1_b": np.asarray(s1_bs).reshape(batch_size, max_len),
                      "s2_f": np.asarray(s2_fs).reshape(batch_size, max_len),
                      "s2_b": np.asarray(s2_bs).reshape(batch_size, max_len)})
            y.append(np.asarray(labels).reshape(batch_size, 3))
            s1_fs, s1_bs = [], []
            s2_fs, s2_bs = [], []
            labels = []

        if len(x) == max_records:
            break

    # TODO: for now just drop last batch
    return x, y, stats

def accuracy(confusion):
    # ratio of on diagonal vs not on diagonal
    return np.sum(confusion * np.identity(len(confusion))) / np.sum(confusion)

def dts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def mean_sd(v):
    if len(v) == 0:
        return None
    else:
        return {"mean": float(np.mean(v)), "sd": float(np.std(v))}

def optimizer(opts):
    lr = float(opts.learning_rate)
    assert lr > 0, "need to set --learning-rate (>0)"
    if opts.optimizer == "GradientDescentOptimizer":
        return tf.train.GradientDescentOptimizer(lr)
    elif opts.optimizer == "AdamOptimizer":
        return tf.train.AdamOptimizer(learning_rate=lr)
    elif opts.optimizer == "MomentumOptimizer":
        m = float(opts.momentum)
        assert m > 0, "MomentumOptimizer requires --momentum (>0)"
        return tf.train.MomentumOptimizer(learning_rate=lr, momentum=m)
    else:
        raise Exception("unknown optimizer [%s]" % opts.optimizer)


