# Consts for mongo
import os

ENTRY_TITLE = "title"
ENTRY_CATEGORIZES = "categories"
ENTRY_PID = "pageID"
ENTRY_TEXT = "text"
ENTRY_LINKS = "links"
CSV_SEPARATOR = "\t"
LINKS_SEPARATOR = ";"

CLEAN_WIKI = "cleanwiki"
PAGES = "pages"
WIKI_LANG = os.environ.get("WIKISEARCH_LANG") or "simplewiki"
EMBEDDINGS = "embeddings"
