# Read in raw data and clean up the column names
import pandas as pd
import re
import string
import nltk

import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer


class TextProcessing():
    stopwords = nltk.corpus.stopwords.words('english')
    wiki_embeddings = api.load('glove-wiki-gigaword-100')

    # remove punctuation in our messages
    def remove_punct(text):
        text = "".join([char for char in text if char not in string.punctuation])
        return text

    # split sentences into a list of words
    def tokenize(text):
        tokens = re.split('\W+', text)
        return tokens

    # remove all stopwords
    def remove_stopwords(tokenized_text):    
        text = [word for word in tokenized_text if word not in stopwords]
        return text

    # to handle all data cleaning
    def clean_text(text):
        text = "".join([word.lower() for word in text if word not in string.punctuation])
        tokens = re.split('\W+', text)
        text = [word for word in tokens if word not in stopwords]
        return text

    def tfidf_vect(func, df):
        # fit TFIDF
        tfidf_vect = TfidfVectorizer(analyzer=func)
        # get results
        X_tfidf = tfidf_vect.fit_transform(df['text'])
        # convert sparse matrix into a df
        X_features = pd.DataFrame(X_tfidf.toarray())
        return X_features

    def most_similar(text):
        return wiki_embeddings.most_similar(text)

    # convert documents to vectors
    def doc_to_vec(docs, search, vector_size, window, min_count):
        tagged_docs_tr = [gensim.models.doc2vec.TaggedDocument(v, [i]) for i, v in enumerate(docs)]

        d2v_model = gensim.models.Doc2Vec(tagged_docs_tr,
                                        vector_size=vector_size,
                                        window=window,
                                        min_count=min_count)
        return d2v_model.infer_vector(search)

        