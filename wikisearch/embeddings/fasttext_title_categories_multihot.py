import torch

from wikisearch.consts.mongo import ENTRY_TITLE
from wikisearch.embeddings import FastTextTitle, CategoriesMultiHot


class FastTextTitleCategoriesMultiHot(FastTextTitle):
    def __init__(self, save_to_db=True):
        # We do not save the FastTextTitleCategoriesMultiHot to db because of size of embeddings in MongoDB
        super(FastTextTitleCategoriesMultiHot, self).__init__(save_to_db=False)
        self._fasttext_title_embedder = FastTextTitle(save_to_db)
        self._categories_multihot_embedder = CategoriesMultiHot(save_to_db)

        # Because embeddings are not in the DB, and we want to save time during runtime, we build all concatenated
        # embeddings up-front
        fasttext_embedder_titles = set(self._fasttext_title_embedder._cached_embeddings.keys())
        categories_embedder_titles = set(self._categories_multihot_embedder._cached_embeddings.keys())
        self._cached_embeddings = {title: torch.cat((self._fasttext_title_embedder.embed(title),
                                                     self._categories_multihot_embedder.embed(title)), dim=0)
                                   for title in fasttext_embedder_titles & categories_embedder_titles}

        # Clear caches of sub-embedders, to free up memory, because we've already got those cached entried
        # concatenated in the current embedder
        self._fasttext_title_embedder._cached_embeddings = {}
        self._categories_multihot_embedder._cached_embeddings = {}

    def _embed(self, page):
        return torch.cat((self._fasttext_title_embedder.embed(page[ENTRY_TITLE]),
                          self._categories_multihot_embedder.embed(page[ENTRY_TITLE])), dim=0)
