from wikisearch.embeddings.embedding import Embedding


class FastText(Embedding):
    def __init__(self):
        super(FastText, self).__init__()

    def embed(self, title):
        self._cache(2, title, 2)


if __name__ == "__main__":
    embedding = FastText()
    embedding.embed("Apple")
