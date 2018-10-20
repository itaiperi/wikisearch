import datetime
import pickle
from abc import ABCMeta, abstractmethod


from wikisearch.consts.mongo import WIKI_LANG, EMBEDDINGS, PAGES, ENTRY_TEXT, ENTRY_ID
from wikisearch.utils.mongo_handler import MongoHandler


class Embedding(metaclass=ABCMeta):
    def __init__(self, database, collection):
        self._database = database
        self._collection = collection
        self._mongo_handler = MongoHandler(database, collection)

    def _load_page(self, title):
        return self._mongo_handler.get_page(WIKI_LANG, EMBEDDINGS, title)

    def _store(self, page_id, title, tensor):
        page = {'_id': page_id, 'title': title, self.__class__.__name__.lower():  self._encode_tensor(tensor),
                'last_modified': datetime.datetime.now().__str__()}
        self._mongo_handler.update_page(WIKI_LANG, EMBEDDINGS, page)

    def embed(self, title):
        page = self._load_page(title)
        if page:
            return page[self.__class__.__name__.lower()]

        page = self._mongo_handler.get_page(WIKI_LANG, PAGES, title)
        text = page[ENTRY_TEXT]

        tokenized_text = self.tokenize_text(text)
        embedded_text_vector = self._embed(tokenized_text)

        self._store(page[ENTRY_ID], title, embedded_text_vector)
        return embedded_text_vector

    @staticmethod
    @abstractmethod
    def tokenize_text(title):
        raise NotImplementedError

    @staticmethod
    def _encode_tensor(tensor):
        return pickle.dumps(tensor)

    @abstractmethod
    def _embed(self, tokenized_text):
        raise NotImplementedError
