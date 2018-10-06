from wikisearch.graph_node import GraphNode
from wikisearch.utils.consts import *
from wikisearch.utils.mongo_handler import MongoHandler


class WikiGraph(dict):
    def __init__(self, wiki_lang):
        super(WikiGraph, self).__init__()
        self._mongo_handler = MongoHandler(wiki_lang, PAGES)

        for entry in self._mongo_handler.get_all_pages():
            title, pid, text, links = entry[ENTRY_TITLE], int(entry[ENTRY_PID]), \
                                      entry[ENTRY_TEXT], entry[ENTRY_LINKS]
            if title in self:
                raise ValueError(f"More than 1 entry with title: '{title}'")
            self[title] = GraphNode(title, pid, text, links)

    def get_node(self, title):
        return self.get(title, None)

    def get_node_neighbors(self, node):
        for link in node.get_neighbors():
            node = self.get_node(link)
            if node:
                yield node
