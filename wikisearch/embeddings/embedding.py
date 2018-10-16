from abc import ABCMeta, abstractmethod

from wikisearch.consts.mongo import WIKI_LANG, PAGES, EMBEDDINGS
from wikisearch.utils.mongo_handler import MongoHandler


class Embedding(metaclass=ABCMeta):
    def __init__(self):
        self._mongo_handler = MongoHandler(WIKI_LANG, PAGES)

    def _cache(self, page_id, title, tensor):
        document = {'_id': page_id, 'title': title, self.__class__.__name__.lower():  tensor}
        self._mongo_handler.update_a_document(WIKI_LANG, EMBEDDINGS, title, document)

    @abstractmethod
    def embed(self, title):
        # TODO: get the _id from the wiki database pages collection (keep on consistency), and pass along to _cache
        raise NotImplementedError
