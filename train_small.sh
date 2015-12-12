#!/usr/bin/env bash
set -x
rm -r train_small
mkdir train_small
./nn_train.py \
--train-set=data/snli_1.0_train.jsonl --num-from-train=10 \
--dev-set=data/snli_1.0_train.jsonl --num-from-dev=10 \
--dev-run-freq=1 --batch-size=2 --num-epochs=5 \
--optimizer=AdamOptimizer \
--embedding-dim=5 --hidden-dim=5 \
--mlp-config=[] --disable-gpu \
--output-vocab-file=train_small/vocab.train.tsv \
--ckpt-dir=train_small

