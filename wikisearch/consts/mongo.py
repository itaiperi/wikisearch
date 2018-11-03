import os

# Consts for mongo
ENTRY_ID = '_id'
ENTRY_TITLE = "title"
ENTRY_CATEGORIZES = "categories"
ENTRY_PID = "pageID"
ENTRY_TEXT = "text"
ENTRY_LINKS = "links"
LINKS_SEPARATOR = ";"

CLEAN_WIKI = "cleanwiki"
PAGES = "pages"
WIKI_LANG = os.environ.get("WIKISEARCH_LANG") or "simplewiki"
EMBEDDINGS = "embeddings"

CSV_SEPARATOR = "\t"
