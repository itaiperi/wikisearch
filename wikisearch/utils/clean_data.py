import os
import string
import time

from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from pymongo import MongoClient

# from consts import *
from wikisearch.utils.consts import *


def tokenize_text(text):
    # Splits dashed words
    clean_text = text.replace('-', ' ')
    # Splits to tokens
    tokens = word_tokenize(clean_text)
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
    tokens = [word for word in tokens if word]
    return tokens


def tokenize_links(links):
    # Filters out external wikipedia links titles
    external_links_titles = {"wikt"}
    processed_links = [link for link in links if link not in external_links_titles]
    wikt_links = ":wikt:"
    processed_links = [link for link in processed_links if wikt_links not in link]
    # Tokenizes each link title
    processed_links = [tokenize_text(link) for link in processed_links]
    # Clears empty links lists
    processed_links = [link for link in processed_links if link]
    return processed_links


class CleanData:
    def __init__(self, wiki_lang):
        self._connection = MongoClient()
        self._db = self._connection.get_database(wiki_lang)
        self._pages_collection = self._db.get_collection("pages")

        start = time.time()
        with open("wikisearch/utils/clean_data.csv", "w", encoding='utf-8') as f_entries:
            for entry in self._pages_collection.find({"title": "People's Republic of China"}):
                title, pid, url, text, links = tokenize_text(entry[ENTRY_TITLE]), entry[ENTRY_PID], None, \
                                                tokenize_text(entry[ENTRY_TEXT]), tokenize_links(entry[ENTRY_LINKS])
                # TODO Need to check if we need to remove stuff like paris trips etc.

                # Write entry to CSV
                f_entries.write(f"{ENTRY_TITLE}\n{title}\n\n")
                f_entries.write(f"{ENTRY_TEXT}\n{str(text)}\n\n")
                f_entries.write(
                    f"-----------------------------------------------------------------------------------------\n\n")

                f_entries.write(f"{ENTRY_LINKS}\n")
                f_entries.write(f"{str(links)}\n")

        print(f"Processing took {int(time.time() - start)} seconds")


if __name__ == "__main__":
    wiki_lang_env = os.environ.get("WIKISEARCH_LANG") or "simplewiki"
    CleanData(wiki_lang_env)
