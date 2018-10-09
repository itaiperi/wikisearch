from wikisearch.embeddings.embedding import Embedding


class Word2Vec(Embedding):
    def __init__(self):
        super(Word2Vec, self).__init__()

    def embed(self, title):
        self._cache(2, title, 1)


if __name__ == "__main__":
    embedding = Word2Vec()
    embedding.embed("Apple")
