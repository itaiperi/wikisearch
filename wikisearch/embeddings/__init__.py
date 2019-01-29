from .embedding import Embedding
from .doc2vec import Doc2Vec
from .word2vec import Word2Vec
from .word2vec_text_average import Word2VecTextAverage
from .fasttext import FastText
from .fasttext_text_average import FastTextTextAverage

# AVAILABLE_EMBEDDINGS - names of classes
AVAILABLE_EMBEDDINGS = [cls.__name__ for cls in [Doc2Vec, Word2VecTextAverage, FastTextTextAverage]]
# EMBEDDINGS_MODULES - names of files in which the AVAILABLE_MODULES are at, respectively.
EMBEDDINGS_MODULES = {
    Doc2Vec.__name__: 'doc2vec',
    Word2Vec.__name__: 'word2vec',
    Word2VecTextAverage.__name__: 'word2vec_average',
    FastText.__name__: 'fasttext',
    FastTextTextAverage.__name__: 'fasttext_average'
}
