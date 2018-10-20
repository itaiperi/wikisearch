import torch

from wikisearch.embeddings import Word2Vec


class Word2VecAverage(Word2Vec):
    def _embed(self, tokenized_text):
        embedded_words = [self._model[tagged_word] for tagged_word in tokenized_text
                          if tagged_word in self._model.__dict__['vocab']]

        # Don't delete!!!!!!
        # Getting also the words without a vector representation
        # embedded_words = []
        # missing_vector_words = []
        # for word in tokenized_text:
        #     if word in self._model.__dict__['vocab']:
        #         embedded_words.append(self._model[word])
        #     else:
        #         missing_vector_words.append(word)

        torched_words_vectors = torch.Tensor(embedded_words)

        return torch.mean(torched_words_vectors, 0)

