import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell

class BidirGruConcatMlp(object):

    def __init__(self):
        self.inited = False

    def add_args(self, parser):
        parser.add_argument("--seq-len", default=25, type=int,
                            help="hack; need to do bucketing, for now just ignore long egs")
        parser.add_argument('--hidden-dim', default=100, type=int,
                            help='hidden node dimensionality')
        parser.add_argument('--embedding-dim', default=100, type=int,
                            help='embedding node dimensionality')
        parser.add_argument("--mlp-config", default="[35]",
                            help="pre classifier mlp config; array describing #hidden nodes"
                                 + " per layer. eg [50,50,20] denotes 3 hidden layers, with 50, 50 and 20"
                                 + " nodes. a value of [] denotes no MLP before classifier")
        parser.add_argument('--initial-embeddings',
                            help='initial embeddings npy file. requires --vocab-file')
        parser.add_argument('--dont-train-embeddings', action='store_true',
                            help='if set don\'t backprop to embeddings')

    def input_placeholders(self, opts):
        self.s1_f = tf.placeholder(tf.int32, [opts.batch_size, opts.seq_len])  # forward over s1
        self.s1_b = tf.placeholder(tf.int32, [opts.batch_size, opts.seq_len])  # backwards over s1
        self.s2_f = tf.placeholder(tf.int32, [opts.batch_size, opts.seq_len])
        self.s2_b = tf.placeholder(tf.int32, [opts.batch_size, opts.seq_len])
        return self.s1_f, self.s1_b, self.s2_f, self.s2_b

    def logits(self, vocab, opts):
        # embed each of the s1/s2 f/b passes
        # for now using the same embedding matrix
        with tf.device("/cpu:0"):
            vs, ed = vocab.size(), opts.embedding_dim
            embeddings = tf.Variable(tf.random_normal([vs, ed]), name="embeddings")
            if opts.initial_embeddings:
                e = np.load(opts.initial_embeddings)
                assert e.shape[0] == vocab.size(), "pretrained embeddings size (%s) != vocab size (%s)" % (e.shape[0], vocab.size())
                assert e.shape[1] == opts.embedding_dim, "pretrained embedding dim (%s) != configured embedding dim (%s)" % (e.shape[1], opts.embedding_dim)
                embeddings.assign(e)
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
            if opts.dont_train_embeddings:
                embedded_inputs = tf.stop_gradient(embedded_inputs)
            # unpack on seq_len dimension (1) into an array of len seq_len
            inputs = tf.split(1, opts.seq_len, embedded_inputs)  # [(batch_size, 1, embedding_dim), ...]
            # squeeze each element (ie throw away "empty dimension"). collapses from 3d to 2d
            inputs = [tf.squeeze(i) for i in inputs]  # [(batch_size, embedding_dim), ...]
            return inputs
        inputs = [self.s1_f, self.s1_b, self.s2_f, self.s2_b]
        embedded_inputs = [embedded_sequence(s) for s in inputs]

        # build an rnn over each sequence capturing the final state from each.
        def final_state_of_rnn_over_embedded_sequence(idx, embedded_seq):
            with tf.variable_scope("rnn_%s" % idx):
                gru = rnn_cell.GRUCell(opts.hidden_dim)
                initial_state = gru.zero_state(opts.batch_size, tf.float32)
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
        last_layer_size = opts.hidden_dim * 4
        for n, num_hidden_nodes in enumerate(eval(opts.mlp_config)):
            last_layer = mlp_layer("mlp_hidden_%d" % n, last_layer, last_layer_size, num_hidden_nodes)
            last_layer_size = num_hidden_nodes
        logits = mlp_layer("logits", last_layer, last_layer_size, 3, include_nonlinearity=False)
        predicted = tf.nn.softmax(logits)
        return logits, predicted
