import datetime
import pickle
import time
from abc import ABCMeta, abstractmethod

import torch

from wikisearch.consts.embeddings import EMBEDDING_VECTOR_SIZE
from wikisearch.consts.mongo import WIKI_LANG, ENTRY_ID, PAGES, ENTRY_TITLE, ENTRY_EMBEDDING
from wikisearch.utils.mongo_handler import MongoHandler


class Embedding(metaclass=ABCMeta):
    """
    Base class for representing an embedding type
    """

    def __init__(self, save_to_db=True, db_prefix=""):
        start = time.time()
        self.save_to_db = save_to_db
        self.type = self.__class__.__name__
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._mongo_handler_pages = MongoHandler(WIKI_LANG, PAGES)
        self._mongo_handler_embeddings = MongoHandler(WIKI_LANG, self.type.lower() + db_prefix)
        self._cached_embeddings = {doc[ENTRY_TITLE]: self._decode_vector(doc[ENTRY_EMBEDDING])
                                   for doc in self._mongo_handler_embeddings.get_all_documents()}
        print(f"-TIME- Took {time.time() - start:2f}s to load {self.__class__.__name__} embedder")

    def _load_embedding(self, title):
        """
        Loads the title's embedding from the database. If doesn't exist returns None
        :param title: the title to search its embedding in the database
        :return: the title's embedding, or None if doesn't exist
        """
        # Check if vector is cached in memory
        vector = self._cached_embeddings.get(title)
        if vector is not None:
            # Don't need to send to device, because _decode_vector already does it
            return vector
        page = self._mongo_handler_embeddings.get_page(title, {"title": True, ENTRY_EMBEDDING: True})
        # TODO what happens if page is None? should this worry us? raise exception?
        if page:
            vector = page.get(ENTRY_EMBEDDING)
            if vector is not None:
                # Don't need to send to device, because _decode_vector already does it
                decoded_vector = self._decode_vector(vector)
                # Cache in memory, for next use
                self._cached_embeddings[title] = decoded_vector
                return decoded_vector

    def _store_embedding(self, page_id, title, vector):
        """
        Stores the title embedding in the database.
        :param page_id: The title's id in the original database
        :param title: The titles to keep its embedding
        :param vector: The title's embedding
        """
        if title not in self._cached_embeddings:
            self._cached_embeddings[title] = vector
        if self.save_to_db:
            page = {'_id': page_id, ENTRY_TITLE: title, ENTRY_EMBEDDING: self._encode_vector(vector),
                    'last_modified': datetime.datetime.now().__str__()}
            self._mongo_handler_embeddings.update_page(page)

    def embed(self, title):
        """
        Embeds the title's text
        :param title: The title to embed its text
        :return: The embedded title's text
        """
        vector = self._load_embedding(title)
        if vector is not None:
            return vector.to(self._device)

        page = self._mongo_handler_pages.get_page(title)
        embedded_vector = self._embed(page)

        self._store_embedding(page[ENTRY_ID], title, embedded_vector)
        return embedded_vector.to(self._device)

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
        return pickle.loads(vector)

    @abstractmethod
    def _embed(self, page):
        """
        The embedding method uniquely per embedding type
        :param page: The text after it was tokenized
        """
        raise NotImplementedError

    def _zeros_if_empty_vector(self, vector):
        return vector if len(vector.size()) else torch.zeros(sum(EMBEDDING_VECTOR_SIZE[self.type].values()), dtype=torch.float)
