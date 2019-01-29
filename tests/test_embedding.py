import time

from wikisearch.consts.mongo import WIKI_LANG, PAGES
from wikisearch.embeddings import FastTextAverage, Word2VecAverage


# Before running this test upload a coreNLPServer (from cmd and not from power shell!)
def test_word2vec_average_flow_embedding():
    embedding = Word2VecAverage(WIKI_LANG, PAGES)
    _test_embedding_timing(embedding)


def test_fasttext_average_flow_embedding():
    embedding = FastTextAverage(WIKI_LANG, PAGES)
    _test_embedding_timing(embedding)


def test_empty_vectors_recognition():
    embedding_fasttext = FastTextAverage(WIKI_LANG, PAGES)
    embedding_word2vec = Word2VecAverage(WIKI_LANG, PAGES)
    title = "July 12"
    embedding_vector_fasttext = embedding_fasttext.embed(title)
    embedding_vector_word2vec = embedding_word2vec.embed(title)
    print(f"FastText"
          f"Is the vector empty?: {'Yes' if not embedding_vector_fasttext.size() else 'No'}\n"
          f"And by the other method: {'Yes' if embedding_vector_fasttext.nelement() == 0 else 'No'}\n")
    print(f"Word2Vec"
          f"Is the vector empty?: {'Yes' if not embedding_vector_word2vec.size() else 'No'}\n"
          f"And by the other method: {'Yes' if embedding_vector_word2vec.nelement() == 0 else 'No'}\n")


def _test_embedding_timing(embedding):
    start = time.time()
    title = "Lorde"
    embedding.embed(title)
    end = time.time()
    print(f"-INFO- Took to embed '{title}' text: {end-start}s.\n"
          f"-INFO- Took {(end-start)/663}s per word.")
    start = time.time()
    title = "People's Republic of China"
    embedding.embed(title)
    end = time.time()
    print(f"-INFO- Took to embed '{title}' text: {end-start}s.\n"
          f"-INFO- Took {(end-start)/2570}s per word")


if __name__ == "__main__":
    # test_word2vec_average_flow_embedding()
    # test_fasttext_average_flow_embedding()
    test_empty_vectors_recognition()
