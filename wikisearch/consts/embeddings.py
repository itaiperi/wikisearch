import os

from wikisearch.consts.mongo import WIKI_LANG, CATEGORIES
from wikisearch.utils.mongo_handler import MongoHandler

_categories_mongo = MongoHandler(WIKI_LANG, CATEGORIES)
categories_multihot_size = len(_categories_mongo.get_page(CATEGORIES)[CATEGORIES])

K_MEANS = os.environ.get("WIKISEARCH_K_MEANS") or 5

EMBEDDING_VECTOR_SIZE = {
    "FastTextTitle": {"embed_dim": 300},
    "FastTextTextAverage": {"embed_dim": 300},
    "FastTextTextKMeans": {"embed_dim": 300 * K_MEANS, "k_means": K_MEANS},
    "Word2VecTitle": {"embed_dim": 300},
    "Word2VecTextAverage": {"embed_dim": 300},
    "Word2VecTextKMeans": {"embed_dim": 300 * K_MEANS, "k_means": K_MEANS},
    "CategoriesMultiHot": {"categories_dim": categories_multihot_size},
    "FastTextTitleCategoriesMultiHot": {"embed_dim": 300, "categories_dim": categories_multihot_size},
    "Word2VecTitleCategoriesMultiHot": {"embed_dim": 300, "categories_dim": categories_multihot_size},
}
