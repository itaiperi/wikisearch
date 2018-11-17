import argparse
import pickle
import re
import time

from wikisearch.consts.paths import PATH_TO_REDIRECT_LOOKUP_TABLE
from wikisearch.utils.mongo_handler import MongoHandler
from wikisearch.consts.mongo import *


def get_nonexistent_wikipedia_pages(wiki_language):
    start = time.time()
    mongo_handler = MongoHandler(wiki_language, PAGES)
    titles = {entry[ENTRY_TITLE] for entry in mongo_handler.project_page_by_field(ENTRY_TITLE)}
    links = {link for entry in mongo_handler.project_page_by_field(ENTRY_LINKS) for link in entry[ENTRY_LINKS]}

    with open(os.path.join("wikisearch", "data_for_visibility", "nonexistent_wikipedia_pages"), 'w', encoding='utf-8') \
            as nonexistent_pages_file:
        nonexistent_pages = links - (links & titles)
        for page in nonexistent_pages:
            nonexistent_pages_file.write(f"{page}\n")

        print(f"-TIME- Getting the nonexistent pages took: {int(time.time() - start)}s")
        return nonexistent_pages


def get_redirect_pages():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xml', required=True)

    args = parser.parse_args()

    redirect_re = re.compile("#REDIRECT \[\[(.*)]]")
    title_re = re.compile("<title>(.*)</title>")

    redirect_pages = {}

    with open(args.xml, 'r', encoding='utf-8') as file:
        title = None
        for line in file.readlines():
            title_matches = title_re.findall(line)
            if title_matches:
                title = title_matches[0]
            else:
                redirect_matches = redirect_re.findall(line)
                if redirect_matches:
                    redirect = redirect_matches[0]
                    redirect_pages[title] = redirect

    with open(os.path.join("wikisearch", "data_for_visibility", "redirect_wikipedia_pages"), 'w', encoding='utf-8') \
            as redirect_pages_file:
        for page, redirect_page in redirect_pages.items():
            redirect_pages_file.write(f"{page} -> {redirect_page}\n")

    print(f"-TIME- Getting the redirect pages took: {int(time.time() - start)}s")
    return redirect_pages


def build_redirect_pages_lookup_table():
    start = time.time()
    nonexistent_pages = get_nonexistent_wikipedia_pages(WIKI_LANG)
    redirect_pages = get_redirect_pages()
    redirect_pages_set = set(redirect_pages.keys())

    nonexistent_because_are_redirect_pages = redirect_pages_set & nonexistent_pages
    lookup_table_for_redirect_pages = {}
    for page in nonexistent_because_are_redirect_pages:
        lookup_table_for_redirect_pages[page] = redirect_pages[page]

    with open(os.path.join("wikisearch", "data_for_visibility", "redirect_lookup_table"), 'w', encoding='utf8') as file:
        for page, redirect_page in lookup_table_for_redirect_pages.items():
            file.write(f"{page} -> {redirect_page}\n")

    with open(PATH_TO_REDIRECT_LOOKUP_TABLE, 'wb') as file:
        pickle.dump(lookup_table_for_redirect_pages, file, protocol=-1)

    print(f"-TIME- Building the redirect lookup table took: {int(time.time() - start)}s")


if __name__ == "__main__":
    build_redirect_pages_lookup_table()
