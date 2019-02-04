from wikisearch.consts.mongo import *
from wikisearch.graph_node import GraphNode
from wikisearch.utils.mongo_handler import MongoHandler


class WikiGraph(dict):
    """
    A graph containing all the wikipedia pages as nodes
    """

    def __init__(self):
        super(WikiGraph, self).__init__()
        self._mongo_handler = MongoHandler(WIKI_LANG, PAGES)
        self._redirects = {}

        for entry in self._mongo_handler.get_all_documents():
            # Handle redirections by inserting into redirects dict
            if ENTRY_REDIRECT_TO in entry:
                self._redirects[entry[ENTRY_TITLE]] = entry[ENTRY_REDIRECT_TO]
            # Handle "normal" entries
            else:
                title, pid, text, links = entry[ENTRY_TITLE], int(entry[ENTRY_PID]), \
                                          entry[ENTRY_TEXT], entry[ENTRY_LINKS]
                if title in self:
                    raise ValueError(f"More than 1 entry with title: '{title}'")
                self[title] = GraphNode(title, pid, text, links)

    def get_node(self, title):
        """
        Gets the wikipedia page with the given title
        :param title: The wikipedia page title to return
        :return: The wikipedia page with the given title, or None if doesn't exist
        """
        # If no node with the title exists, maybe it's a redirect, so we try to return that.
        return self.get(title) or self.get(self._redirects.get(title))

    def get_node_neighbors(self, node):
        """
        Gets the node's neighbors if the node exists in the graph. Otherwise, returns None
        :param node: The node to returns its neighbors
        """
        for link in node.neighbors:
            node = self.get_node(link)
            if node:
                yield node
