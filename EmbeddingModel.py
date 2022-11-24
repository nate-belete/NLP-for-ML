import tensorflow as tf
 
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
       
    # get target word and the left and right widnow size
   def get_target_and_size(self, sequence, target_index, window_size):
        # target word
        target_word = sequence[target_index]
        # window size
        half_window_size = window_size // 2

        return target_word, half_window_size

    # Return the left (inclusive) and right (exclusive) indices of the window.
   def  get_window_indices(self, sequence, target_index, half_window_size):
        left_incl = max(0,target_index - half_window_size)
        right_incl = min(len(sequence), target_index + half_window_size + 1)
        return left_incl, right_incl

    # Convert a list of text strings into word sequences
   def get_target_and_context(self, sequence, target_index, window_size):
        target_word, half_window_size = self.get_target_and_size(sequence, target_index, window_size)
        left_incl, right_excl = self.get_window_indices(sequence, target_index, half_window_size)
        return target_word, left_incl, right_excl