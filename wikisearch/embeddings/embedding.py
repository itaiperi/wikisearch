import datetime
import pickle
from abc import ABCMeta, abstractmethod

from wikisearch.consts.mongo import WIKI_LANG, EMBEDDINGS
from wikisearch.utils.mongo_handler import MongoHandler


class Embedding(metaclass=ABCMeta):
    def __init__(self, database, collection):
        self._database = database
        self._collection = collection
        self._mongo_handler = MongoHandler(database, collection)

    @abstractmethod
    def embed(self, title):
        raise NotImplementedError

    def _load_page(self, title):
        return self._mongo_handler.get_page(WIKI_LANG, EMBEDDINGS, title)

    def _store(self, page_id, title, tensor):
        page = {'_id': page_id, 'title': title, self.__class__.__name__.lower():  self._encode_tensor(tensor),
                'last_modified': datetime.datetime.now().__str__()}
        self._mongo_handler.update_page(WIKI_LANG, EMBEDDINGS, page)

    @staticmethod
    def _encode_tensor(tensor):
        return pickle.dumps(tensor)
