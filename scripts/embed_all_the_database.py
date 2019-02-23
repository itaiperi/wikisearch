import argparse
import datetime
import time

from scripts.loaders import load_embedder_by_name
from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import WIKI_LANG, PAGES, ENTRY_TITLE, ENTRY_ID
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS
from wikisearch.utils.mongo_handler import MongoHandler


def get_update_embedding_request(page_pr):
    page_title = page_pr[ENTRY_TITLE]

    embedded_vector = embedder._load_embedding(page_title)
    if embedded_vector is None:
        embedded_vector = embedder._embed(page_pr)

    page_id = page_pr[ENTRY_ID]

    return embedder._mongo_handler_embeddings.update_page_request(
        {'_id': page_id, 'title': page_title,
         embedder.__class__.__name__.lower(): embedder._encode_vector(embedded_vector),
         'last_modified': datetime.datetime.now().__str__()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True, choices=AVAILABLE_EMBEDDINGS)
    args = parser.parse_args()

    embedder = load_embedder_by_name(args.embedding)
    mongo_handler_pages = MongoHandler(WIKI_LANG, PAGES)

    pages = mongo_handler_pages.get_all_documents()
    len_pages = pages.count()
    start = time.time()
    update_embeddings_requests = []
    for idx, page in enumerate(pages, 1):
        update_embeddings_requests.append(get_update_embedding_request(page))
        print_progress_bar(idx, len_pages, time.time() - start, prefix=f'Embedding', length=50)

    embedder._mongo_handler_embeddings.bulk_write(update_embeddings_requests)

    print(f"-TIME- Took to embed all the database: {time.time()-start}")
