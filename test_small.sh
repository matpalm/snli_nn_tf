#!/usr/bin/env bash
set -x
./nn_test.py \
--test-set=data/snli_1.0_train.jsonl --num-from-test=10 \
--batch-size=2 --embedding-dim=5 --hidden-dim=5 \
--mlp-config=[] --disable-gpu \
--vocab-file=train_small/vocab.train.tsv \
--restore-ckpt=$1
