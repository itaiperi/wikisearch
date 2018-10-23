import time

from wikisearch.consts.mongo import WIKI_LANG, PAGES
from wikisearch.embeddings import FastTextAverage, Word2VecAverage


# Before running this test upload a coreNLPServer (from cmd and not from power shell!)
def test_word2vec_flow_embedding():
    embedding = Word2VecAverage(WIKI_LANG, PAGES)
    start = time.time()
    embedding.embed("Lorde")
    end = time.time()
    print(f"Took to embed 'Lorde' text: {end-start}s.\n"
          f"Took {(end-start)/663}s per word.")
    start = time.time()
    embedding.embed("People's Republic of China")
    end = time.time()
    print(f"Took to embed 'People's Republic of China' text: {end-start}s.\n"
          f"Took {(end-start)/2570}s per word")


def test_fasttext_flow_embedding():
    embedding = FastTextAverage(WIKI_LANG, PAGES)
    start = time.time()
    embedding.embed("Lorde")
    end = time.time()
    print(f"Took to embed 'Lorde' text: {end-start}s.\n"
          f"Took {(end-start)/663}s per word.")
    start = time.time()
    embedding.embed("People's Republic of China")
    end = time.time()
    print(f"Took to embed 'People's Republic of China' text: {end-start}s.\n"
          f"Took {(end-start)/2570}s per word")


if __name__ == "__main__":
    # Before running one of the tests change the path of 'PATH_TO_PRETRAINED_MODEL'
    # to the location of the wanted pretrained model

    test_word2vec_flow_embedding()
    # test_fasttext_flow_embedding()
