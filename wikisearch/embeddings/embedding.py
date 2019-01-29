import datetime
import pickle
from abc import ABCMeta, abstractmethod

import torch

from wikisearch.consts.mongo import WIKI_LANG, EMBEDDINGS, PAGES, ENTRY_ID
from wikisearch.consts.nn import EMBEDDING_VECTOR_SIZE
from wikisearch.utils.mongo_handler import MongoHandler


class Embedding(metaclass=ABCMeta):
    """
    Base class for representing an embedding type
    """

    def __init__(self, database, collection):
        self._database = database
        self._collection = collection
        self._mongo_handler = MongoHandler(database, collection)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._type = self.__class__.__name__

    def _load_embedding(self, title):
        """
        Loads the title's embedding from the database. If doesn't exist returns None
        :param title: the title to search its embedding in the database
        :return: the title's embedding, or None if doesn't exist
        """
        page = self._mongo_handler.get_page(WIKI_LANG, EMBEDDINGS, title)
        if page:
            vector = page.get(self.__class__.__name__.lower())
            return self._decode_vector(vector) if vector else None

    def _store(self, page_id, title, vector):
        """
        Stores the title embedding in the database.
        :param page_id: The title's id in the original database
        :param title: The titles to keep its embedding
        :param vector: The title's embedding
        """
        page = {'_id': page_id, 'title': title, self.__class__.__name__.lower(): self._encode_vector(vector),
                'last_modified': datetime.datetime.now().__str__()}
        self._mongo_handler.update_page(WIKI_LANG, EMBEDDINGS, page)

    def embed(self, title):
        """
        Embeds the title's text
        :param title: The title to embed its text
        :return: The embedded title's text
        """
        vector = self._load_embedding(title)
        if vector is not None:
            return vector.to(self._device)

        page = self._mongo_handler.get_page(WIKI_LANG, PAGES, title)
        embedded_vector = self._embed(page)

        # TODO: remove once the wtf-wikipedia parse lists correctly
        if len(embedded_vector.size()) == 0:
            embedded_vector = torch.zeros(EMBEDDING_VECTOR_SIZE).to(self._device)

        self._store(page[ENTRY_ID], title, embedded_vector)
        return embedded_vector.to(self._device)

    @staticmethod
    @abstractmethod
    def tokenize_text(title):
        """
        Tokenizes the title's text by the embedding class
        :param title: The title to tokenize its text
        """
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self):
        """
        Returns metadata relevant to the embedding
        :return: dictionary with key-metadata pairs.
            Compulsory keys:
                * type - name of embedding (fasttext, word2vec, etc.)
                * vectors_filepath - path to weights vectors file that is used
        """
        raise NotImplementedError

    @staticmethod
    def _encode_vector(vector):
        """
        Encodes the vector to a saveable value in the database
        :param vector: The vector to encode
        :return: The encoded vector
        """
        return pickle.dumps(vector)

    def _decode_vector(self, vector):
        """
        Decodes the vector's value in the database to its original representation
        :param vector: The vector to decode
        :return: The decoded vector
        """
        return pickle.loads(vector).to(self._device)

    @abstractmethod
    def _embed(self, page):
        """
        The embedding method uniquely per embedding type
        :param page: The text after it was tokenized
        """
        raise NotImplementedError
