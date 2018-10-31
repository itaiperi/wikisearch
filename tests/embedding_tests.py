import time

from wikisearch.consts.mongo import WIKI_LANG, PAGES
from wikisearch.embeddings import FastTextAverage, Word2VecAverage


# Before running this test upload a coreNLPServer (from cmd and not from power shell!)
def test_word2vec_average_flow_embedding():
    print("----------------Testing Word2VecAverage--------------")
    embedding = Word2VecAverage(WIKI_LANG, PAGES)
    _test_embedding_timing(embedding)


def test_fasttext_average_flow_embedding():
    print("----------------Testing FastTextAverage--------------")
    embedding = FastTextAverage(WIKI_LANG, PAGES)
    _test_embedding_timing(embedding)


def test_empty_vectors_recognition():
    print("--------------------Testing empty vectors recognition----------------")
    embedding_fasttext = FastTextAverage(WIKI_LANG, PAGES)
    embedding_word2vec = Word2VecAverage(WIKI_LANG, PAGES)
    title = "Long-term memory"
    embedding_vector_fasttext = embedding_fasttext.embed(title)
    embedding_vector_word2vec = embedding_word2vec.embed(title)
    print(f"FastText"
          f"Is the vector empty?: {'Yes' if embedding_vector_fasttext.size() else 'No'}\n"
          f"And by the other method: {'Yes' if embedding_vector_fasttext.nelement() == 0 else 'No'}\n"
          f"By the third: {'Yes' if len(embedding_vector_fasttext)==0 else 'No'}")
    print(f"Word2Vec"
          f"Is the vector empty?: {'Yes' if embedding_vector_word2vec.size() else 'No'}\n"
          f"And by the other method: {'Yes' if embedding_vector_word2vec.nelement() == 0 else 'No'}\n"
          f"By the third: {'Yes' if len(embedding_vector_word2vec)==0 else 'No'}")


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
