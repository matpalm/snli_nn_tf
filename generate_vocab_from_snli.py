#!/usr/bin/env python
from collections import Counter
import json
import sys
from tokenise_parse import *

tokens = Counter()
for line in sys.stdin:
    d = json.loads(line)
    tokens.update(tokens_for(d, 1, 'PARSE_WITH_OPEN_CLOSE_TAGS'))
    tokens.update(tokens_for(d, 2, 'PARSE_WITH_OPEN_CLOSE_TAGS'))

sys.stdout.write("PAD\t0\nUNK\t1\n")
idx = 2
for token, _freq in tokens.most_common():
    sys.stdout.write("%s\t%s\n" % (token, idx))
    idx += 1

