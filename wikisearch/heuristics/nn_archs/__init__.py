from wikisearch.heuristics.nn_archs.text_kmeans_categories_multihot_distance import TextKMeansCategoriesMultiHotDistance
from wikisearch.heuristics.nn_archs.text_kmeans_distance import TextKMeansDistance
from wikisearch.heuristics.nn_archs.title_distance import TitleDistance
from wikisearch.heuristics.nn_archs.title_text_kmeans_categories_multihot_distance import \
    TitleTextKMeansCategoriesMultiHotDistance
from wikisearch.heuristics.nn_archs.title_text_kmeans_distance import TitleTextKMeansDistance

NN_ARCHS = [cls.__name__ for cls in [TitleTextKMeansCategoriesMultiHotDistance, TitleDistance, TextKMeansDistance,
                                     TextKMeansCategoriesMultiHotDistance, TitleTextKMeansDistance]]
