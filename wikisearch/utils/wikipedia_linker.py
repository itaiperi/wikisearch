import os

from wikisearch.utils.mongo_handler import MongoHandler
from wikisearch.utils.consts import *


class WikipediaLinker:
    @staticmethod
    def get_unexist_wikipedia_pages(wiki_language):
        mongo_handler = MongoHandler(wiki_language, PAGES)
        titles = {entry[ENTRY_TITLE].lower() for entry in mongo_handler.project_page_by_field(ENTRY_TITLE)}
        links = {link.lower() for entry in mongo_handler.project_page_by_field(ENTRY_LINKS) for link in entry[ENTRY_LINKS]}

        with open(os.path.join("wikisearch", "utils", "unexist_wikipedia_pages"), 'w', encoding='utf8') as unexist_pages_file:
            unexist_pages = links - (links & titles)
            for page in unexist_pages:
                unexist_pages_file.write(f"{page}\n")


if __name__ == "__main__":
    wiki_lang_env = os.environ.get("WIKISEARCH_LANG") or "simplewiki"
    WikipediaLinker.get_unexist_wikipedia_pages(wiki_lang_env)
