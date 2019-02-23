from wikisearch.consts.mongo import WIKI_LANG, PAGES, ENTRY_CATEGORIES, ENTRY_REDIRECT_TO, CATEGORIES, ENTRY_TITLE
from wikisearch.utils.mongo_handler import MongoHandler


def main():
    pages_mongo = MongoHandler(WIKI_LANG, PAGES)
    categories_mongo = MongoHandler(WIKI_LANG, CATEGORIES)

    categories = set()
    for doc in pages_mongo.get_all_documents():
        if ENTRY_REDIRECT_TO not in doc:
            categories.update(doc[ENTRY_CATEGORIES])

    categories_mongo.update_page({ENTRY_TITLE: CATEGORIES, CATEGORIES: sorted(categories)})
    print(f"Found {len(categories)} categories")


if __name__ == "__main__":
    main()
