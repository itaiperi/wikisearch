from wikisearch.consts.mongo import WIKI_LANG, CATEGORIES
from wikisearch.utils.mongo_handler import MongoHandler

_categories_mongo = MongoHandler(WIKI_LANG, CATEGORIES)
categories_multihot_size = len(_categories_mongo.get_page(CATEGORIES)[CATEGORIES])

EMBEDDING_VECTOR_SIZE = {
    "FastTextTitle": 300,
    "FastTextTextAverage": 300,
    "Word2VecTitle": 300,
    "Word2VecTextAverage": 300,
    "CategoriesMultiHot": categories_multihot_size,
    "FastTextTitleCategoriesMultiHot": 300 + categories_multihot_size,
    "Word2VecTitleCategoriesMultiHot": 300 + categories_multihot_size,
}
