from wikisearch.consts.mongo import PAGES, WIKI_LANG
from wikisearch.embeddings.embedding import Embedding


class FastText(Embedding):
    def __init__(self):
        super(FastText, self).__init__(WIKI_LANG, PAGES)

    def embed(self, title):
        pass
