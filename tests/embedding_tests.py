import time

from wikisearch.consts.mongo import WIKI_LANG, PAGES
from wikisearch.embeddings import FastTextAverage


def test_flow_embedding():
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
    test_flow_embedding()
