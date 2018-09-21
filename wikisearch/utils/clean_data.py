import os
import string
import time
from multiprocessing.pool import Pool

from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from pymongo import MongoClient

from wikisearch.utils.consts import *


def tokenize_text(text):
    # TODO Need to check if we need to remove stuff like paris trips etc.
    # Splits dashed words
    text = text.replace('-', ' ')
    # Splits to tokens
    tokens = word_tokenize(text)
    # Converts each word to lower-case
    tokens = [word.lower() for word in tokens]
    # Removes punctuation from each word
    table_of_punctuation = str.maketrans('', '', string.punctuation)
    tokens = [word.translate(table_of_punctuation) for word in tokens]
    # Removes non-alpha words
    # tokens = [word for word in tokens if word.isalpha()]
    # Filters out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stems the words
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    # Clears empty words
    return [word for word in tokens if word]


def tokenize_links(links):
    # Filters out external wikipedia links titles
    external_links_titles = {"wikt"}
    processed_links = [link for link in links if link not in external_links_titles]
    # TODO: change later to regex of :.+: - check it first in mongo to see which get hit
    wikt_links = ":wikt:"
    processed_links = [link for link in processed_links if wikt_links not in link]
    # processed_links = [link.lower() for link in processed_links]
    return processed_links


def tokenize_title(title):
    # return title.lower()
    return title


def clean_page(entry):
    # print(f"Cleaning: {entry[ENTRY_TITLE]}")
    return {ENTRY_TITLE: tokenize_title(entry[ENTRY_TITLE]),
            ENTRY_CATEGORIZES: entry[ENTRY_CATEGORIZES],
            ENTRY_TEXT: tokenize_text(entry[ENTRY_TEXT]),
            ENTRY_LINKS: tokenize_links(entry[ENTRY_LINKS]),
            ENTRY_PID: entry[ENTRY_PID]}


class CleanData:
    def __init__(self, wiki_lang):
        self._mongo_client = MongoClient()
        self._pages_collection = self._mongo_client.get_database(wiki_lang).get_collection(PAGES)

        start = time.time()

        db_names = self._mongo_client.list_database_names()
        if CLEAN_WIKI in db_names:
            self._mongo_client.drop_database(CLEAN_WIKI)
        clean_wiki = self._mongo_client[CLEAN_WIKI][PAGES]

        with Pool(6) as pool:
            clean_pages = list(pool.map(clean_page, self._pages_collection.find({})))

        clean_wiki.insert_many(clean_pages)
        print(f"Cleaning took {int(time.time() - start)} seconds")


if __name__ == "__main__":
    wiki_lang_env = os.environ.get("WIKISEARCH_LANG") or "simplewiki"
    CleanData(wiki_lang_env)
