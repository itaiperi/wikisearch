import os

from pymongo import MongoClient

from wikisearch.graph_node import GraphNode
from wikisearch.utils.consts import ENTRY_TITLE, ENTRY_PID, ENTRY_TEXT, ENTRY_LINKS

wiki_lang = os.environ.get("WIKISEARCH_LANG") or "simplewiki"
connection = MongoClient()
db = connection.get_database(wiki_lang)
entries = db.get_collection("pages")


class WikiGraph(dict):
    def __init__(self):
        super(WikiGraph, self).__init__()
        wiki_lang = os.environ.get("WIKISEARCH_LANG") or "simplewiki"
        self.connection = MongoClient()
        self.db = connection.get_database(wiki_lang)
        self.entries = db.get_collection("pages")

        for entry in self.entries.find({}):
            title, pid, url, text, links = entry[ENTRY_TITLE], int(entry[ENTRY_PID]), None, entry[ENTRY_TEXT], entry[ENTRY_LINKS]
            if title in self:
                raise ValueError("More than 1 entry with title", title)
            self[title] = GraphNode(title, pid, url, text, links)

    def get_node(self, title):
        return self.get(title, None)

    def get_node_neighbors(self, node):
        for link in node.get_neighbors():
            node = self.get_node(link)
            if node:
                yield self.get_node(link)
