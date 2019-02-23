import torch

from wikisearch.consts.mongo import WIKI_LANG, CATEGORIES
from wikisearch.embeddings import Embedding
from wikisearch.utils.mongo_handler import MongoHandler


class CategoriesMultiHot(Embedding):
    def __init__(self):
        super(CategoriesMultiHot, self).__init__()
        categories_mongo = MongoHandler(WIKI_LANG, CATEGORIES)
        categories = categories_mongo.get_all_documents()[0][CATEGORIES]
        self._categories_map = {category: i for i, category in enumerate(categories)}

    def get_metadata(self):
        """
        Returns metadata relevant to the embedding
        :return: dictionary with key-metadata pairs.
            Compulsory keys:
                * type - name of embedding (fasttext, word2vec, etc.)
        """
        return {
            'type': self._type,
        }

    def _embed(self, page):
        vector = torch.zeros(len(self._categories_map), dtype=torch.long)
        page_categories_indices = [self._categories_map[category] for category in page[CATEGORIES]]
        vector.scatter_(0, torch.Tensor(page_categories_indices).long(), 1)
        return vector
