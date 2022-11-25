import tensorflow as tf



# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    def get_input_target_sequence(self, sequence):
        seq_len = len(sequence)
        if seq_len >= self.max_length:
            input_sequence = sequence[:self.max_length - 1]
            target_sequence = sequence[1:self.max_length]

        else:
            padding_amount = self.max_length - len(sequence)
            padding = [0] * padding_amount
            input_sequence = sequence[:-1] + padding
            target_sequence = sequence[1:] + padding

        return input_sequence, target_sequence