from .categories_multihot import CategoriesMultiHot
from .doc2vec import Doc2Vec
from .embedding import Embedding
from .fasttext import FastText
from .fasttext_text_average import FastTextTextAverage
from .fasttext_text_kmeans import FastTextTextKMeans
from .fasttext_title import FastTextTitle
from .fasttext_title_categories_multihot import FastTextTitleCategoriesMultiHot
from .word2vec import Word2Vec
from .word2vec_text_average import Word2VecTextAverage
from .word2vec_title import Word2VecTitle
from .word2vec_title_categories_multihot import Word2VecTitleCategoriesMultiHot

# AVAILABLE_EMBEDDINGS - names of classes
AVAILABLE_EMBEDDINGS = \
    [cls.__name__ for cls in [Doc2Vec,
                              FastTextTitle, FastTextTextAverage, FastTextTextKMeans, FastTextTitleCategoriesMultiHot,
                              Word2VecTitle, Word2VecTextAverage, Word2VecTitleCategoriesMultiHot,
                              CategoriesMultiHot]]
# EMBEDDINGS_MODULES - names of files in which the AVAILABLE_MODULES are at, respectively.
EMBEDDINGS_MODULES = {
    Doc2Vec.__name__: 'doc2vec',
    Word2Vec.__name__: 'word2vec',
    Word2VecTitle.__name__: 'word2vec_title',
    Word2VecTextAverage.__name__: 'word2vec_text_average',
    FastText.__name__: 'fasttext',
    FastTextTitle.__name__: 'fasttext_title',
    FastTextTextAverage.__name__: 'fasttext_text_average',
    FastTextTextKMeans.__name__: 'fasttext_text_k_means',
    CategoriesMultiHot.__name__: 'categories_multihot',
    FastTextTitleCategoriesMultiHot.__name__: 'fasttext_title_categories_multihot',
    Word2VecTitleCategoriesMultiHot.__name__: 'word2vec_title_categories_multihot',
}
