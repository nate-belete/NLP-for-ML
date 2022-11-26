import tensorflow as tf
import numpy as np

# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    # Create a cell for the LSTM
    def make_lstm_cell(self, dropout_keep_prob):
        cell = tf.keras.layers.LSTMCell(self.num_lstm_units, dropout=dropout_keep_prob)
        return cell

    # Stack multiple layers for the LSTM
    def stacked_lstm_cells(self, is_training):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob) for i in range(self.num_lstm_layers)]
        cell = tf.keras.layers.StackedRNNCells(cell_list)
        return cell_list

     # Convert input sequences to embeddings
    def get_input_embeddings(self, input_sequences):
        embedding_dim = int(self.vocab_size**0.25)
        embedding=tf.keras.layers.Embedding(
            self.vocab_size+1, embedding_dim, embeddings_initializer='uniform',
            mask_zero=True, input_length=self.max_length
        )
        input_embeddings = embedding(input_sequences)
        return input_embeddings

    # Run the LSTM on the input sequences
    def run_lstm(self, input_sequences, is_training):
        cell = self.stacked_lstm_cells(is_training)
        input_embeddings = self.get_input_embeddings(input_sequences)
        binary_sequences = tf.math.sign(input_sequences)
        sequence_lengths = tf.math.reduce_sum(binary_sequences, axis=1)
        rnn=tf.keras.layers.RNN(
            cell,
            return_sequences=True,
            input_length=sequence_lengths,
            dtype=tf.float32
        )
        lstm_outputs = rnn(input_embeddings)
        return lstm_outputs, binary_sequences

    # calculate the loss function 
    def calculate_loss(self, lstm_outputs, binary_sequences, output_sequences):
        logits = tf.keras.layers.Dense(self.vocab_size)(lstm_outputs)
        batch_sequence_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=output_sequences, logits=logits)
        unpadded_loss = batch_sequence_loss * tf.cast(binary_sequences, tf.float32)
        overall_loss = tf.math.reduce_sum(unpadded_loss)
        return overall_loss

    # Predict next word ID
    def get_word_predictions(self, word_preds, binary_sequences, batch_size):
        row_indices = tf.range(batch_size)
        final_indexes = tf.math.reduce_sum(binary_sequences, axis=1) - 1
        gather_indices = tf.transpose([row_indices, final_indexes])
        final_id_predictions = tf.gather_nd(word_preds, gather_indices)
        return final_id_predictions
