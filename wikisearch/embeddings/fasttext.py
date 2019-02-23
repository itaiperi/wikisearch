import string
import time
from abc import ABC

import gensim
from nltk import word_tokenize
from nltk.corpus import stopwords

from scripts.utils import Cache
from wikisearch.consts.paths import PATH_TO_PRETRAINED_FASTTEXT_MODEL

from wikisearch.embeddings.embedding import Embedding


class FastText(Embedding, ABC):
    """
    The class representing the fasttext embedding and its derivatives
    """

    def __init__(self):
        """
        Load the embedding pre-trained model
        """
        super(FastText, self).__init__()
        cache = Cache()
        start = time.time()
        self._model = cache['fasttext_model']
        if not self._model:
            self._model = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_PRETRAINED_FASTTEXT_MODEL)
            cache['fasttext_model'] = self._model
        print(f"-TIME- Took {time.time() - start}s to load the pretrained model")

    @staticmethod
    def tokenize_text(text):
        """
        Tokenizes the title's text by the embedding class
        :param title: The title to tokenize its text
        """
        # Filters out external links
        text = ' '.join([word for word in text.split() if "https://" not in word and "http://" not in word])
        # Tokenize the text
        text = word_tokenize(text)
        # Removes stop words
        stop_words = set(stopwords.words('english'))
        # Removes punctuation from each word
        punctuation = set(string.punctuation) | {"\"\""} | {'\'\''} | {'``'}
        punctuation_and_stop_words = stop_words | punctuation
        text = [word for word in text if word not in punctuation_and_stop_words]
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
            'type': self.type,
            'vectors_filepath': PATH_TO_PRETRAINED_FASTTEXT_MODEL,
        }
