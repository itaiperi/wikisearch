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

    def _load_embedding(self, title):
        page = self._mongo_handler.get_page(WIKI_LANG, EMBEDDINGS, title)
        if page:
            vector = page.get(self.__class__.__name__.lower())
            return Embedding._decode_tensor(vector) if vector else None

    def _store(self, page_id, title, tensor):
        page = {'_id': page_id, 'title': title, self.__class__.__name__.lower(): Embedding._encode_tensor(tensor),
                'last_modified': datetime.datetime.now().__str__()}
        self._mongo_handler.update_page(WIKI_LANG, EMBEDDINGS, page)

    def embed(self, title):
        vector = self._load_embedding(title)
        if vector is not None:
            return vector

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

    @staticmethod
    def _decode_tensor(tensor):
        return pickle.loads(tensor)

    @abstractmethod
    def _embed(self, tokenized_text):
        raise NotImplementedError
