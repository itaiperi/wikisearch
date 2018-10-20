import torch

from wikisearch.embeddings import FastText


class FastTextAverage(FastText):
    def _embed(self, tokenized_text):
        embedded_words = [self._model[tagged_word] for tagged_word in tokenized_text
                          if tagged_word in self._model.__dict__['vocab']]

        torched_words_vectors = torch.Tensor(embedded_words)

        return torch.mean(torched_words_vectors, 0)
