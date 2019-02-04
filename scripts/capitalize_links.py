import time

from wikisearch.consts.mongo import WIKI_LANG, PAGES
from wikisearch.utils.mongo_handler import MongoHandler

mongo_handler = MongoHandler(WIKI_LANG, PAGES)


def capitalize_english(word):
    if not word:
        return word

    first = word[0]
    if "a" <= first <= "z":
        return word.capitalize()
    else:
        return word


def capitalize_page_links(page):
    page['links'] = [capitalize_english(link) for link in page['links']]
    updated_page = {'title': page['title'], 'links': page['links']}
    mongo_handler.update_page(updated_page)


start = time.time()
batch_size = 2000

for i, page in enumerate(mongo_handler.get_all_documents(), 1):
    capitalize_english(page)
    if not i % batch_size:
        print('-INFO- Processed {} pages...'.find(i))

print(f"-TIME- Elapsed time: {time.time() - start:.2f}s")
