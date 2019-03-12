import argparse
import datetime
import time

from scripts.loaders import load_embedder_by_name
from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import WIKI_LANG, PAGES, ENTRY_TITLE, ENTRY_ID, ENTRY_REDIRECT_TO, ENTRY_EMBEDDING
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS
from wikisearch.utils.mongo_handler import MongoHandler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, choices=AVAILABLE_EMBEDDINGS, nargs="+")
    parser.add_argument("-b", "--batch", default=1000, type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    mongo_handler_pages = MongoHandler(WIKI_LANG, PAGES)
    for embedding in args.embeddings:
        embedder_name = embedding.lower()
        # Embeddings' collecion name is the embedding name, in lowercase
        mongo_handler_embeddings = MongoHandler(WIKI_LANG, embedder_name)
        if args.overwrite:
            mongo_handler_embeddings.delete_collection_data()
        # set index, if there isn't one in place
        mongo_handler_embeddings.create_title_index()

        embedded_pages_query = {ENTRY_EMBEDDING: {"$exists": True}}
        embedded_titles = [page[ENTRY_TITLE] for page in mongo_handler_embeddings.get_pages(embedded_pages_query, {ENTRY_TITLE: True})]

        embedder = load_embedder_by_name(embedding, save_to_db=False)
        # Only embed pages which are not embedded yet, and which are not redirects
        pages_without_embeddings_query = {"title": {"$nin": embedded_titles},
                                          ENTRY_REDIRECT_TO: {"$exists": False}}
        pages = mongo_handler_pages.get_pages(pages_without_embeddings_query)
        len_pages = pages.count()

        start = time.time()
        embedded_vectors = []
        for idx, page in enumerate(pages, 1):

            page_value = {'_id': page[ENTRY_ID], ENTRY_TITLE: page[ENTRY_TITLE],
                          'last_modified': datetime.datetime.now().__str__(),
                          ENTRY_EMBEDDING: embedder._encode_vector(embedder._embed(page))}

            embedded_vectors.append(page_value)
            if idx == len_pages or idx % args.batch == 0:
                mongo_handler_embeddings.insert_data(embedded_vectors)
                embedded_vectors = []

            print_progress_bar(idx, len_pages, time.time() - start, prefix=f'Embedding {embedding}', length=50)
