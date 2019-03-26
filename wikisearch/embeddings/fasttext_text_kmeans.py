import numpy as np
import torch

from wikisearch.consts.embeddings import KMEANS, EMBEDDING_VECTOR_SIZE
from wikisearch.consts.mongo import ENTRY_TEXT
from wikisearch.embeddings import FastText

from sklearn.cluster import k_means


class FastTextTextKMeans(FastText):
    """
    The class represents the fasttext embedding when a page's vector is calculated by taking the K mean vectors
    of all the words in the page's text
    """

    def __init__(self, save_to_db=True):
        super(FastTextTextKMeans, self).__init__(save_to_db=save_to_db, db_prefix=str(KMEANS))

    def _embed(self, page):
        tokenized_text = self.tokenize_text(page[ENTRY_TEXT])
        embedded_words = [self._model[tagged_word] for tagged_word in tokenized_text
                          if tagged_word in self._model.__dict__['vocab']]

        if len(embedded_words) > KMEANS:
            mean_vectors, _, _ = k_means(embedded_words, KMEANS, n_init=10)
            # Flatten vectors to one long vector
            return torch.from_numpy(mean_vectors.reshape([-1])).float()

        else:
            mean_vectors = np.concatenate(embedded_words, axis=0) if len(embedded_words) else np.zeros(EMBEDDING_VECTOR_SIZE[self.type]["embed_dim"])
            mean_vectors = np.pad(mean_vectors, pad_width=(0, EMBEDDING_VECTOR_SIZE[self.type]["embed_dim"] - mean_vectors.size),
                                  mode="constant", constant_values=0)
            return torch.from_numpy(mean_vectors).float()
