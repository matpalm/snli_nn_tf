from collections import Counter, defaultdict
import json
import numpy as np
import random
import sys
import time
import tokenise_parse

LABELS = ['contradiction', 'neutral', 'entailment']

def label_for(eg):
    try:
        return LABELS.index(eg['gold_label'])
    except ValueError:
        return None

def accuracy(confusion):
    # ratio of on diagonal vs not on diagonal
    return np.sum(confusion * np.identity(len(confusion))) / np.sum(confusion)

