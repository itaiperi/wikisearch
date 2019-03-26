import os

from wikisearch.consts.mongo import WIKI_LANG, CATEGORIES
from wikisearch.utils.mongo_handler import MongoHandler

_FASTTEXT = 300
_WORD2VEC = 300
_categories_mongo = MongoHandler(WIKI_LANG, CATEGORIES)
_CATEGORIES_MULTIHOT = len(_categories_mongo.get_page(CATEGORIES)[CATEGORIES])
KMEANS = int(os.environ.get("WIKISEARCH_K_MEANS")) or 5

EMBEDDING_VECTOR_SIZE = {
    "FastTextTitle": {"embed_dim": _FASTTEXT},
    "FastTextTextAverage": {"embed_dim": _FASTTEXT},
    "FastTextTextKMeans": {"embed_dim": _FASTTEXT * KMEANS, "k_means": KMEANS},
    "FastTextTitleTextKMeans": {"embed_dim": _FASTTEXT + _FASTTEXT * KMEANS},
    "FastTextTextKMeansCategoriesMultiHot": {"embed_dim": _FASTTEXT * KMEANS, "categories_dim": _CATEGORIES_MULTIHOT},
    "FastTextTitleTextKMeansCategoriesMultiHot": {"embed_dim": _FASTTEXT + _FASTTEXT * KMEANS,
                                                  "categories_dim": _CATEGORIES_MULTIHOT},
    "Word2VecTitle": {"embed_dim": _WORD2VEC},
    "Word2VecTextAverage": {"embed_dim": _WORD2VEC},
    "Word2VecTextKMeans": {"embed_dim": _WORD2VEC * KMEANS, "k_means": KMEANS},
    "Word2VecTitleTextKMeans": {"embed_dim": _WORD2VEC + _WORD2VEC * KMEANS},
    "Word2VecTextKMeansCategoriesMultiHot": {"embed_dim": _WORD2VEC * KMEANS, "categories_dim": _CATEGORIES_MULTIHOT},
    "Word2VecTitleTextKMeansCategoriesMultiHot": {"embed_dim": _WORD2VEC + _WORD2VEC * KMEANS,
                                                  "categories_dim": _CATEGORIES_MULTIHOT},
}
