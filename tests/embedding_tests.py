import time

from wikisearch.consts.mongo import WIKI_LANG, PAGES
from wikisearch.embeddings import FastTextAverage, Word2VecAverage


# Before running this test upload a coreNLPServer (from cmd and not from power shell!)
def test_word2vec_average_flow_embedding():
    print("----------------Testing Word2VecAverage--------------")
    embedding = Word2VecAverage(WIKI_LANG, PAGES)
    start = time.time()
    embedding.embed("Lorde")
    end = time.time()
    print(f"-INFO- Took to embed 'Lorde' text: {end-start}s.\n"
          f"-INFO- Took {(end-start)/663}s per word.")
    start = time.time()
    embedding.embed("People's Republic of China")
    end = time.time()
    print(f"-INFO- Took to embed 'People's Republic of China' text: {end-start}s.\n"
          f"-INFO- Took {(end-start)/2570}s per word")


def test_fasttext_average_flow_embedding():
    print("----------------Testing FastTextAverage--------------")
    embedding = FastTextAverage(WIKI_LANG, PAGES)
    start = time.time()
    embedding.embed("Lorde")
    end = time.time()
    print(f"-INFO- Took to embed 'Lorde' text: {end-start}s.\n"
          f"-INFO- Took {(end-start)/663}s per word.")
    start = time.time()
    embedding.embed("People's Republic of China")
    end = time.time()
    print(f"-INFO- Took to embed 'People's Republic of China' text: {end-start}s.\n"
          f"-INFO- Took {(end-start)/2570}s per word")


if __name__ == "__main__":
    test_word2vec_average_flow_embedding()
    # test_fasttext_average_flow_embedding()
