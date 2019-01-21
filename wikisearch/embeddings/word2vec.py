import string
from abc import ABC

import gensim
import corenlp
import time

from nltk.corpus import stopwords

from scripts.utils import Cache
from wikisearch.consts.paths import PATH_TO_PRETRAINED_WORD2VEC_MODEL
from wikisearch.consts.pos_conversion import TREEBANK_TO_UNIVERSAL
from wikisearch.embeddings import Embedding


class Word2Vec(Embedding, ABC):
    """
    The class representing the word2vec embedding and its derivatives
    """

    def __init__(self, database, collection):
        """
        Load the embedding pre-trained model
        :param database: The database to connect to this instance
        :param collection: The collection to connect to this instance
        """
        super(Word2Vec, self).__init__(database, collection)
        cache = Cache()
        start = time.time()
        self._model = cache['word2vec_model']
        if not self._model:
            self._model = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_PRETRAINED_WORD2VEC_MODEL)
            cache['word2vec_model'] = self._model
        print(f"-TIME- Took {time.time() - start:.1f}s to load the pretrained model")

    @staticmethod
    def tokenize_text(text):
        # Filters out external links
        text = ' '.join([word for word in text.split() if "https://" not in word and "http://" not in word])
        # Start the coreNLPServer separately
        with corenlp.CoreNLPClient(start_server=False, timeout=10000, annotators="tokenize ssplit lemma pos".split()) as client:
            ann = client.annotate(text)
        # Filters out stop words
        stop_words = set(stopwords.words('english'))
        # Removes punctuation from each word
        punctuation = set(string.punctuation) | {"\"\""} | {'\'\''} | {'``'}
        punctuation_and_stop_words = stop_words | punctuation
        # Couple each word with its pos tag if the word isn't a stop word and isn't a punctuation
        text = [f"{token.lemma}_{TREEBANK_TO_UNIVERSAL[token.pos]}" for sentence in ann.sentence
                for token in sentence.token if token.lemma not in punctuation_and_stop_words]
        return text

    def get_metadata(self):
        """
        Returns metadata relevant to the embedding
        :return: dictionary with key-metadata pairs.
            Compulsory keys:
                * type - name of embedding (fasttext, word2vec, etc.)
                * vectors_filepath - path to weights vectors file that is used
        """
        return {
            'type': 'word2vec',
            'vectors_filepath': PATH_TO_PRETRAINED_WORD2VEC_MODEL,
        }
