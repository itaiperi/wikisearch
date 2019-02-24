import torch

from wikisearch.embeddings import Word2VecTitle, CategoriesMultiHot


class Word2VecTitleCategoriesMultiHot(Word2VecTitle):
    def __init__(self):
        super(Word2VecTitleCategoriesMultiHot, self).__init__()
        self._word2vec_title_embedder = Word2VecTitle()
        self._categories_multihot_embedder = CategoriesMultiHot()

    def _embed(self, page):
        return torch.cat((self._word2vec_title_embedder._embed(page), self._categories_multihot_embedder._embed(page)),
                         dim=0)
