import os

# Consts for pages collection
ENTRY_ID = '_id'
ENTRY_TITLE = "title"
ENTRY_REDIRECT_TO = "redirectTo"
ENTRY_CATEGORIES = "categories"
ENTRY_PID = "pageID"
ENTRY_TEXT = "text"
ENTRY_LINKS = "links"
LINKS_SEPARATOR = ";"

CLEAN_WIKI = "cleanwiki"
WIKI_LANG = os.environ.get("WIKISEARCH_LANG") or "simplewiki"
PAGES = "pages"
CATEGORIES = "categories"
EMBEDDINGS = "embeddings"

CSV_SEPARATOR = "\t"
