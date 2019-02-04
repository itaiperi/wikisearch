import string

from nltk import word_tokenize
from nltk.corpus import stopwords

from wikisearch.embeddings.embedding import Embedding


class Doc2Vec(Embedding):
    def __init__(self):
        super(Doc2Vec, self).__init__()

    def _embed(self, page):
        # TODO need to implement inference of vectors
        pass

    @staticmethod
    def tokenize_text(text):
        # Splits dashed words
        text = text.replace('–', ' ')
        text = text.replace('-', ' ')
        # Filters out external links
        text = ' '.join([word for word in text.split() if "https://" not in word and "http://" not in word])
        # Splits to tokens
        tokens = word_tokenize(text)
        # Converts each word to lower-case
        tokens = [word.lower() for word in tokens]
        # Removes punctuation, empty and stop words
        stop_words = set(stopwords.words('english'))
        punctuation = set(string.punctuation)
        punc_and_stop_words = stop_words | punctuation
        tokens = [word for word in tokens if (word and word not in punc_and_stop_words)]
        return tokens

    def get_metadata(self):
        """
        Returns metadata relevant to the embedding
        :return: dictionary with key-metadata pairs.
            Compulsory keys:
                * type - name of embedding (fasttext, word2vec, etc.)
                * vectors_filepath - path to weights vectors file that is used
        """
        return {
            'type': 'doc2vec',
            # TODO fill out
            'vectors_filepath': '',
        }
