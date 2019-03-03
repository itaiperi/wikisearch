from wikisearch.consts.mongo import WIKI_LANG, CATEGORIES
from wikisearch.utils.mongo_handler import MongoHandler

_categories_mongo = MongoHandler(WIKI_LANG, CATEGORIES)
categories_multihot_size = len(_categories_mongo.get_page(CATEGORIES)[CATEGORIES])

EMBEDDING_VECTOR_SIZE = {
    "FastTextTitle": {"embed_dim": 300},
    "FastTextTextAverage": {"embed_dim": 300},
    "Word2VecTitle": {"embed_dim": 300},
    "Word2VecTextAverage": {"embed_dim": 300},
    "CategoriesMultiHot": {"categories_dim": categories_multihot_size},
    "FastTextTitleCategoriesMultiHot": {"embed_dim": 300, "categories_dim": categories_multihot_size},
    "Word2VecTitleCategoriesMultiHot": {"embed_dim": 300, "categories_dim": categories_multihot_size},
}
