import argparse
import time

from scripts.loaders import load_embedder_by_name
from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import WIKI_LANG, PAGES, ENTRY_TITLE
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS
from wikisearch.utils.mongo_handler import MongoHandler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True, choices=AVAILABLE_EMBEDDINGS)
    args = parser.parse_args()

    embedder = load_embedder_by_name(args.embedding)
    mongo_handler_pages = MongoHandler(WIKI_LANG, PAGES)

    pages = mongo_handler_pages.get_all_documents()
    len_pages = pages.count()
    start = time.time()
    for idx, page in enumerate(pages, 1):
        embedder.embed(page[ENTRY_TITLE])
        print_progress_bar(idx, len_pages, time.time() - start, prefix=f'Embedding: ', length=50)
