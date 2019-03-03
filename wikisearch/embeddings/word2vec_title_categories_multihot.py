import torch

from wikisearch.consts.mongo import ENTRY_TITLE
from wikisearch.embeddings import Word2VecTitle, CategoriesMultiHot


class Word2VecTitleCategoriesMultiHot(Word2VecTitle):
    def __init__(self, save_to_db=True):
        super(Word2VecTitleCategoriesMultiHot, self).__init__(save_to_db)
        self._word2vec_title_embedder = Word2VecTitle(save_to_db)
        self._categories_multihot_embedder = CategoriesMultiHot(save_to_db)

    def _embed(self, page):
        return torch.cat((self._word2vec_title_embedder.embed(page[ENTRY_TITLE]),
                          self._categories_multihot_embedder.embed(page[ENTRY_TITLE])), dim=0)
