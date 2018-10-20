import time

import torch

from wikisearch.consts.mongo import ENTRY_TEXT, ENTRY_ID, PAGES, WIKI_LANG
from wikisearch.embeddings import Word2Vec


class Word2VecAverage(Word2Vec):
    def embed(self, title):
        document = self._load_document(title)
        if document:
            return document[self.__class__.__name__.lower()]

        page = self._mongo_handler.get_page(WIKI_LANG, PAGES, title)
        text = page[ENTRY_TEXT]
        tokenized_text = Word2Vec.tokenize_text(text)

        embedded_words = [self._model[tagged_word] for tagged_word in tokenized_text
                          if tagged_word in self._model.__dict__['vocab']]

        # Getting also the words without a vector representation
        # embedded_words = []
        # missing_vector_words = []
        # for word in tokenized_text:
        #     if word in self._model.__dict__['vocab']:
        #         embedded_words.append(self._model[word])
        #     else:
        #         missing_vector_words.append(word)

        torched_words_vectors = torch.Tensor(embedded_words)

        mean_vector = torch.mean(torched_words_vectors, 0)
        self._store(page[ENTRY_ID], title, mean_vector)
        return mean_vector
