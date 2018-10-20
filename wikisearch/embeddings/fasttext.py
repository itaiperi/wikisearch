from abc import ABC

from wikisearch.consts.mongo import PAGES, WIKI_LANG
from wikisearch.embeddings.embedding import Embedding


class FastText(Embedding, ABC):
    def __init__(self):
        super(FastText, self).__init__(WIKI_LANG, PAGES)
        self._model = None

    @staticmethod
    def tokenize_text(text):
        pass
