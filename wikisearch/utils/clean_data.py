import string
import time
from multiprocessing.pool import Pool

from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords

from wikisearch.consts.mongo import *
from wikisearch.utils.mongo_handler import MongoHandler


def tokenize_text(text):
    # Splits dashed words
    text = text.replace('-', ' ')
    # Filters out external links
    text = ' '.join([word for word in text.split() if "https://" not in word and "http://" not in word])
    # Splits to tokens
    tokens = word_tokenize(text)
    # Converts each word to lower-case
    tokens = [word.lower() for word in tokens]
    # Removes punctuation, empty and stop words
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    punctuation_and_stop_words = stop_words | punctuation
    tokens = [word for word in tokens if (word and word not in punctuation_and_stop_words)]
    # Stems the words
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    return tokens


def tokenize_links(links):
    # Filters out external wikipedia links titles
    external_links_titles = {"wikt"}
    processed_links = [link for link in links if link not in external_links_titles]
    # TODO: change later to regex of :.+: - check it first in mongo to see which get hit
    wikt_links = ":wikt:"
    processed_links = [link for link in processed_links if wikt_links not in link]
    return processed_links


def tokenize_title(title):
    return title


def clean_page(entry):
    # print(f"Cleaning: {entry[ENTRY_TITLE]}")
    # print(entry[ENTRY_TITLE])
    return {ENTRY_TITLE: tokenize_title(entry[ENTRY_TITLE]),
            ENTRY_CATEGORIZES: entry[ENTRY_CATEGORIZES],
            ENTRY_TEXT: tokenize_text(entry[ENTRY_TEXT]),
            ENTRY_LINKS: tokenize_links(entry[ENTRY_LINKS]),
            ENTRY_PID: entry[ENTRY_PID]}


class CleanData:
    def __init__(self, wiki_lang):
        self._mongo_handler = MongoHandler(wiki_lang, PAGES)

        start = time.time()

        with Pool(4) as pool:
            clean_pages = list(pool.map(clean_page, self._mongo_handler.get_all_pages()))

        self._mongo_handler.create_database_collection_with_data(CLEAN_WIKI, PAGES, clean_pages)
        print(f"Cleaning took {int(time.time() - start)} seconds")


if __name__ == "__main__":
    CleanData(WIKI_LANG)
