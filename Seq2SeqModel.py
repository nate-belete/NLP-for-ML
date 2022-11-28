import tensorflow as tf

# Get c and h vectors for bidirectional LSTM final states
def ref_get_bi_state_parts(state_fw, state_bw):
    bi_state_c = tf.concat([state_fw[0], state_bw[0]], -1)
    bi_state_h = tf.concat([state_fw[1], state_bw[1]], -1)
    return bi_state_c, bi_state_h

# Seq2seq model
class Seq2SeqModel(object):
    def __init__(self, vocab_size, num_lstm_layers, num_lstm_units):
        self.vocab_size = vocab_size
        # Extended vocabulary includes start, stop token
        self.extended_vocab_size = vocab_size + 2
        self.num_lstm_layers = num_lstm_layers
        self.num_lstm_units = num_lstm_units
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size)

    def make_lstm_cell(self, dropout_keep_prob, num_units):
        cell = tf.keras.layers.LSTMCell(num_units, dropout=dropout_keep_prob)
        return cell

    # Create multi-layer LSTM
    def stacked_lstm_cells(self, is_training, num_units):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob, num_units) for i in range(self.num_lstm_layers)]
        cell = tf.keras.layers.StackedRNNCells(cell_list)
        return cell

    # Get embeddings for input/output sequences
    def get_embeddings(self, sequences, scope_name):
        with tf.compat.v1.variable_scope(scope_name,reuse=tf.compat.v1.AUTO_REUSE):
            cat_column = tf.compat.v1.feature_column \
              .categorical_column_with_identity(
                  'sequences', self.extended_vocab_size)
            embed_size = int(self.extended_vocab_size**0.25)
            embedding_column = tf.compat.v1.feature_column.embedding_column(
                  cat_column, embed_size)
            seq_dict = {'sequences': sequences}
            embeddings= tf.compat.v1.feature_column \
                                 .input_layer(
                                     seq_dict, [embedding_column])
            sequence_lengths = tf.compat.v1.placeholder("int64", shape=(None,), name=scope_name+"/sinput_layer/sequence_length")
            return embeddings, tf.cast(sequence_lengths, tf.int32)
    
    # Create the encoder for the model
    def encoder(self, encoder_inputs, is_training):
        input_embeddings, input_seq_lens = self.get_embeddings(encoder_inputs, 'encoder_emb')
        cell = self.stacked_lstm_cells(is_training, self.num_lstm_units)

        combined_state = []
        rnn = tf.keras.layers.RNN(
              cell,
              return_sequences=True,
              return_state=True,
              go_backwards=True,
              dtype=tf.float32)
        Bi_rnn = tf.keras.layers.Bidirectional(
              rnn,
              merge_mode='concat'
              )
        input_embeddings = tf.reshape(input_embeddings, [-1,-1,2])
        outputs = Bi_rnn(input_embeddings)
        enc_outputs = outputs[0]
        states_fw =  [ outputs[i]  for i in range(1,self.num_lstm_layers+1)] 
        states_bw =  [ outputs[i]  for i in range(self.num_lstm_layers+1,len(outputs))]

        for i in range(self.num_lstm_layers):
            bi_state_c, bi_state_h = ref_get_bi_state_parts(
                states_fw[i], states_bw[i]
            )
            bi_lstm_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
            bi_state_c, bi_state_h)
            combined_state.append(bi_lstm_state)
        final_state = tuple(combined_state)    
        return enc_outputs, input_seq_lens, final_state

    # Helper funtion to combine BiLSTM encoder outputs
    def combine_enc_outputs(self, enc_outputs):
        enc_outputs_fw, enc_outputs_bw = enc_outputs
        return tf.concat([enc_outputs_fw, enc_outputs_bw], -1)

    # Create the stacked LSTM cells for the decoder
    def create_decoder_cell(self, enc_outputs, input_seq_lens, is_training):
        num_decode_units = self.num_lstm_units * 2
        dec_cell = self.stacked_lstm_cells(is_training, num_decode_units)
        combined_enc_outputs = self.combine_enc_outputs(enc_outputs)
        attention_mechanism = tfa.seq2seq.LuongAttention(
            num_decode_units, combined_enc_outputs,
            memory_sequence_length=input_seq_lens)
        dec_cell = tfa.seq2seq.AttentionWrapper(
            dec_cell, attention_mechanism,
            attention_layer_size=num_decode_units)
        return dec_cell