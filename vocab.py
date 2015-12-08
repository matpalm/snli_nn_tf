class Vocab(object):

    def __init__(self, vocab_file=None):
        self.token_id = {}
        self.id_token = {}
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.seq = 2
        self.vocab_file = vocab_file
        if vocab_file:
            for line in open(vocab_file, "r"):
                token, idx = line.strip().split("\t")
                idx = int(idx)
                assert token not in self.token_id, "dup entry for token [%s]" % token
                assert idx not in self.id_token, "dup entry for idx [%s]" % idx
                if idx == 0:
                    assert token == "PAD", "expect id 0 to be [PAD] not [%s]" % token
                if idx == 1:
                    assert token == "UNK", "expect id 1 to be [UNK] not [%s]" % token
                self.token_id[token] = idx
                self.id_token[idx] = token

    def size(self):
        return len(self.token_id) + 2  # +1 for UNK & PAD

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
