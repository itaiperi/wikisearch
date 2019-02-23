import torch

from wikisearch.consts.mongo import ENTRY_TITLE
from wikisearch.embeddings import FastText, FastTextTitle, CategoriesMultiHot


class FastTextTitleCategoriesMultiHot(FastText):
    def __init__(self):
        super(FastTextTitleCategoriesMultiHot, self).__init__()
        self._fasttext_title_embedder = FastTextTitle()
        self._categories_multihot_embedder = CategoriesMultiHot()

    def _embed(self, page):
        return torch.cat((self._fasttext_title_embedder._embed(page), self._categories_multihot_embedder._embed(page)),
                         dim=0)
