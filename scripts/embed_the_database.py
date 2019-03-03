import argparse
import datetime
import time

from scripts.loaders import load_embedder_by_name
from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import WIKI_LANG, PAGES, ENTRY_TITLE, ENTRY_ID, ENTRY_REDIRECT_TO, EMBEDDINGS
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS
from wikisearch.utils.mongo_handler import MongoHandler


def get_page_embedding(embedder, page_pr):
    embedded_vector = embedder._embed(page_pr)

    return embedder.__class__.__name__.lower(), embedder._encode_vector(embedded_vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, choices=AVAILABLE_EMBEDDINGS, nargs="+")
    parser.add_argument("-b", "--batch", default=1000, type=int)
    args = parser.parse_args()

    embedders = [load_embedder_by_name(embedder, save_to_db=False) for embedder in args.embeddings]
    mongo_handler_pages = MongoHandler(WIKI_LANG, PAGES)
    mongo_handler_embeddings = MongoHandler(WIKI_LANG, EMBEDDINGS)

    pages = mongo_handler_pages.get_all_documents()
    len_pages = pages.count()
    start = time.time()
    embedded_vectors = []
    for idx, page in enumerate(pages, 1):
        if ENTRY_REDIRECT_TO in page:
            # This is a redirect page, so we don't need to embed it
            continue

        page_value = {'_id': page[ENTRY_ID], 'title': page[ENTRY_TITLE],
                      'last_modified': datetime.datetime.now().__str__()}
        for embedder in embedders:
            embedder_class, embedder_value = get_page_embedding(embedder, page)
            page_value[embedder_class] = embedder_value

        embedded_vectors.append(page_value)
        if idx == len_pages or idx % args.batch == 0:
            mongo_handler_embeddings.insert_data(embedded_vectors)
            embedded_vectors = []

        print_progress_bar(idx, len_pages, time.time() - start, prefix=f'Embedding', length=50)
