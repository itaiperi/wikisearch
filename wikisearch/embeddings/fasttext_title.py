import torch

from wikisearch.consts.mongo import ENTRY_TITLE
from wikisearch.embeddings import FastText


class FastTextTitle(FastText):
    def _embed(self, page):
        tokenized_title = self.tokenize_text(page[ENTRY_TITLE])
        embedded_words = [self._model[tagged_word] for tagged_word in tokenized_title
                          if tagged_word in self._model.__dict__['vocab']]

        torched_words_vectors = torch.Tensor(embedded_words)

        return self._zeros_if_empty_vector(torch.mean(torched_words_vectors, 0))
