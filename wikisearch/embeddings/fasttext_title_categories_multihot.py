import torch

from wikisearch.consts.mongo import ENTRY_TITLE
from wikisearch.embeddings import FastTextTitle, CategoriesMultiHot


class FastTextTitleCategoriesMultiHot(FastTextTitle):
    def __init__(self, save_to_db=True):
        super(FastTextTitleCategoriesMultiHot, self).__init__(save_to_db)
        self._fasttext_title_embedder = FastTextTitle(save_to_db)
        self._categories_multihot_embedder = CategoriesMultiHot(save_to_db)

    def _embed(self, page):
        return torch.cat((self._fasttext_title_embedder.embed(page[ENTRY_TITLE]),
                          self._categories_multihot_embedder.embed(page[ENTRY_TITLE])), dim=0)
