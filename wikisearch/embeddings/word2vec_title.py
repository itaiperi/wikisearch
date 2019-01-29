import torch

from wikisearch.consts.mongo import ENTRY_TITLE
from wikisearch.embeddings import Word2Vec


class Word2VecTitle(Word2Vec):
    def _embed(self, page):
        tokenized_text = self.tokenize_text(page[ENTRY_TITLE])
        embedded_words = [self._model[tagged_word] for tagged_word in tokenized_text
                          if tagged_word in self._model.__dict__['vocab']]

        # Getting also the words without a vector representation
        # embedded_words, missing_vector_words = self._get_embedded_words_and_missing_vectors(tokenized_text)

        torched_words_vectors = torch.Tensor(embedded_words)

        return torch.mean(torched_words_vectors, 0)
