from pymongo import MongoClient

from wikisearch.graph_node import GraphNode
from wikisearch.utils.consts import ENTRY_TITLE, ENTRY_PID, ENTRY_TEXT, ENTRY_LINKS


class WikiGraph(dict):
    def __init__(self, wiki_lang):
        super(WikiGraph, self).__init__()
        self._connection = MongoClient()
        self._db = self._connection.get_database(wiki_lang)
        self._pages_collection = self._db.get_collection("pages")

        for entry in self._pages_collection.find({}):
            title, pid, url, text, links = entry[ENTRY_TITLE], int(entry[ENTRY_PID]), None, entry[ENTRY_TEXT], entry[ENTRY_LINKS]
            if title in self:
                raise ValueError("More than 1 entry with title", title)
            self[title] = GraphNode(title, pid, url, text, links)

    def get_node(self, title):
        return self.get(title, None)

    def get_node_neighbors(self, node):
        for link in node.get_neighbors():
            # Capitalize because some links are not capitalized in the text, but the actual entries are.
            node = self.get_node(link.capitalize())
            if node:
                yield node
