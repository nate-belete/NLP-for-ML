import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
 
# Skip-gram embedding model
class EmbeddingModel(object):
   # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
       self.vocab_size = vocab_size
       self.embedding_dim = embedding_dim
       self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)
 
   # Convert a list of text strings into word sequences
    def tokenize_text_corpus(self, texts):
       # initialize Tokenizer with text
       self.tokenizer.fit_on_texts(texts)
       # convert text corpus into tokenized sequences
       sequences = self.tokenizer.texts_to_sequences(texts)
 
       return sequences

    # Convert a list of text strings into word sequences
    def get_target_and_context(self, sequence, target_index, window_size):
        # target word
        target_word = sequence[target_index]
        # window size
        half_window_size = window_size // 2
        # left window.
        left_incl = max(0, target_index - half_window_size)
        # right window.
        right_excl = min(len(sequence), target_index + half_window_size + 1)
        return target_word, left_incl, right_excl

    # Create (target, context) pairs for a given window size
    def create_target_context_pairs(self, texts, window_size):
        pairs = []
        sequences = self.tokenize_text_corpus(texts)      
        for sequence in sequences:
            for i in range(len(sequence)):
                target_word, left_incl, right_excl = self.get_target_and_context(
                    sequence, i, window_size)
                for j in range(left_incl, right_excl):
                    if j != i:
                        pairs.append((target_word, sequence[j]))
        return pairs

    # Forward run of the embedding model to retrieve embeddings
    def forward(self, target_ids):
        # initialize our embedding matrix variable with a random uniform initializer
        initial_bounds = 0.5 / self.embedding_dim
        initializer = tf.random.uniform((self.vocab_size, self.embedding_dim), minval=-initial_bounds, maxval=initial_bounds)
        
        # create/retrieve the embedding matrix variable, then use it to get embeddings.
        self.embedding_matrix = tf.compat.v1.get_variable('embedding_matrix',initializer=initializer)
        embeddings = tf.compat.v1.nn.embedding_lookup(self.embedding_matrix, target_ids)
        return embeddings

    # Calculate NCE Loss based on the retrieved embedding and context
    def calculate_loss(self, embeddings, context_ids, num_negative_samples):
        weights, bias = self.get_bias_weights()
        nce_losses = tf.nn.nce_loss( weights, bias, context_ids, embeddings, 
                                     num_negative_samples, self.vocab_size)
        overall_loss = tf.math.reduce_mean(nce_losses)
        return overall_loss