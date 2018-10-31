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
    def __init__(self, database, collection):
        super(FastText, self).__init__(database, collection)
        cache = Cache()
        start = time.time()
        self._model = cache['fasttext_model']
        if not self._model:
            self._model = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_PRETRAINED_FASTTEXT_MODEL)
            cache['fasttext_model'] = self._model
        print(f"-INFO- Took {time.time() - start}s to load the pretrained model")

    @staticmethod
    def tokenize_text(text):
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
