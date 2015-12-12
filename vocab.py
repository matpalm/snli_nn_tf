class Vocab(object):

    def __init__(self, vocab_file=None):
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.token_id = {'PAD': self.PAD_ID, 'UNK': self.UNK_ID}
        self.id_token = {v:k for k,v in self.token_id.iteritems()}
        self.seq = len(self.token_id)
        self.vocab_file = vocab_file
        if vocab_file:
            for line in open(vocab_file, "r"):
                token, idx = line.strip().split("\t")
                idx = int(idx)
                if token != 'PAD' and token != 'UNK':
                    assert token not in self.token_id, "dup entry for token [%s]" % token
                    assert idx not in self.id_token, "dup entry for idx [%s]" % idx
                if token == "PAD" or idx == self.PAD_ID:
                    assert token == "PAD" and idx == self.PAD_ID
                if token == "UNK" or idx == self.UNK_ID:
                    assert token == "UNK" and idx == self.UNK_ID
                self.token_id[token] = idx
                self.id_token[idx] = token

    def write_to(self, out_file):
        with open(out_file, "w") as f:
            for tid in sorted(self.id_token.keys()):
                f.write("%s\t%s\n" % (self.id_token[tid], tid))

    def size(self):
        return len(self.token_id) + 2  # UNK + PAD

    def id_for_token(self, token, update=True):
        if token in self.token_id:
            return self.token_id[token]
        elif not update:
            return self.UNK_ID
        elif self.vocab_file is not None:
            raise Exception("cstrd with vocab_file=[%s] but missing entry [%s]" % (self.vocab_file, token))
        else:
            self.token_id[token] = self.seq
            self.id_token[self.seq] = token
            self.seq += 1
            return self.seq - 1

    def ids_for_tokens(self, tokens, update=True):
        return [self.id_for_token(t, update) for t in tokens]
