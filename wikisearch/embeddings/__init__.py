from .categories_multihot import CategoriesMultiHot
from .doc2vec import Doc2Vec
from .embedding import Embedding
from .fasttext import FastText
from .fasttext_text_average import FastTextTextAverage
from .fasttext_text_kmeans import FastTextTextKMeans
from .fasttext_title import FastTextTitle
from .fasttext_title_text_kmeans import FastTextTitleTextKMeans
from .fasttext_text_kmeans_categories_multihot import FastTextTextKMeansCategoriesMultiHot
from .fasttext_title_text_kmeans_categories_multihot import FastTextTitleTextKMeansCategoriesMultiHot
from .word2vec import Word2Vec
from .word2vec_text_average import Word2VecTextAverage
from .word2vec_text_kmeans import Word2VecTextKMeans
from .word2vec_title import Word2VecTitle
from .word2vec_title_text_kmeans import Word2VecTitleTextKMeans
from .word2vec_text_kmeans_categories_multihot import Word2VecTextKMeansCategoriesMultiHot
from .word2vec_title_text_kmeans_categories_multihot import Word2VecTitleTextKMeansCategoriesMultiHot

# AVAILABLE_EMBEDDINGS - names of classes
AVAILABLE_EMBEDDINGS = \
    [cls.__name__ for cls in [Doc2Vec,
                              FastTextTitle, FastTextTextAverage, FastTextTextKMeans, FastTextTitleTextKMeans,
                              FastTextTextKMeansCategoriesMultiHot, FastTextTitleTextKMeansCategoriesMultiHot,
                              Word2VecTitle, Word2VecTextAverage, Word2VecTextKMeans, Word2VecTitleTextKMeans,
                              Word2VecTextKMeansCategoriesMultiHot, Word2VecTitleTextKMeansCategoriesMultiHot,
                              CategoriesMultiHot]]
# EMBEDDINGS_MODULES - names of files in which the AVAILABLE_MODULES are at, respectively.
EMBEDDINGS_MODULES = {
    Doc2Vec.__name__: 'doc2vec',
    FastText.__name__: 'fasttext',
    FastTextTitle.__name__: 'fasttext_title',
    FastTextTextAverage.__name__: 'fasttext_text_average',
    FastTextTextKMeans.__name__: 'fasttext_text_kmeans',
    FastTextTitleTextKMeans.__name__: 'fasttext_text_text_kmeans',
    FastTextTextKMeansCategoriesMultiHot.__name__: 'fasttext_text_kmeans_categories_multihot',
    FastTextTitleTextKMeansCategoriesMultiHot.__name__: 'fasttext_title_text_kmeans_categories_multihot',
    Word2Vec.__name__: 'word2vec',
    Word2VecTitle.__name__: 'word2vec_title',
    Word2VecTextAverage.__name__: 'word2vec_text_average',
    Word2VecTextKMeans.__name__: 'word2vec_text_kmeans',
    Word2VecTitleTextKMeans.__name__: 'word2vec_title_text_kmeans',
    Word2VecTextKMeansCategoriesMultiHot.__name__: 'word2vec_text_kmeans_categories_multihot',
    Word2VecTitleTextKMeansCategoriesMultiHot.__name__: 'word2vec_title_text_kmeans_categories_multihot',
    CategoriesMultiHot.__name__: 'categories_multihot',
}
