from .embedding import Embedding
from .doc2vec import Doc2Vec
from .word2vec import Word2Vec
from .word2vec_average import Word2VecAverage
from .fasttext import FastText
from .fasttext_average import FastTextAverage

# AVAILABLE_EMBEDDINGS - names of classes
AVAILABLE_EMBEDDINGS = ['Doc2Vec', 'Word2VecAverage', 'FastTextAverage']
# EMBEDDINGS_MODULES - names of files in which the AVAILABLE_MODULES are at, respectively.
EMBEDDINGS_MODULES = {
    'Doc2Vec': 'doc2vec',
    'WORD2VEC': 'word2vec',
    'Word2VecAverage': 'word2vec_average',
    'FASTTEXT': 'fasttext',
    'FastTextAverage': 'fasttext_average'
}
