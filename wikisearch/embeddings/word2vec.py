import string
from abc import ABC

import gensim
import corenlp
import time

from nltk.corpus import stopwords

from wikisearch.consts.paths import PATH_TO_PRETRAINED_MODEL
from wikisearch.consts.pos_conversion import TREEBANK_TO_UNIVERSAL
from wikisearch.embeddings import Embedding


class Word2Vec(Embedding, ABC):
    def __init__(self, database, collection):
        start = time.time()
        super(Word2Vec, self).__init__(database, collection)
        self._model = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_PRETRAINED_MODEL, binary=False)
        print(f"Took: {time.time() - start:.1f}s to load the pretrained model")

    @staticmethod
    def tokenize_text(text):
        # Filters out external links
        text = [word for word in text.split() if "https://" not in word and "http://" not in word]
        text = ' '.join(text)
        # Start the coreNLPServer separately
        with corenlp.CoreNLPClient(start_server=False, timeout=10000, annotators="tokenize ssplit lemma pos".split()) as client:
            ann = client.annotate(text)
        # Filters out stop words
        stop_words = set(stopwords.words('english'))
        # Removes punctuation from each word
        table_of_punctuation = str.maketrans('', '', string.punctuation)
        # Couple each word with its pos tag if the word isn't a stop word and isn't a punctuation
        text = [f"{token.lemma}_{TREEBANK_TO_UNIVERSAL[token.pos]}" for sentence in ann.sentence
                for token in sentence.token
                if token.lemma not in stop_words and token.word.translate(table_of_punctuation)]
        return text
