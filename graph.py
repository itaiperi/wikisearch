from graph_node import GraphNode
from utils.consts import *

class WikiGraph(dict):
    def __init__(self, csv_filepath):
        super()
        with open(csv_filepath) as csv_file:
            for line in csv_file.readlines():
                title, pid, url, text, links = line.split(CSV_SEPARATOR)
                # print(pid, url, text, links, links.split())
                self[title] = GraphNode(title, pid, url, text, links.split(LINKS_SEPARATOR))

    def get_node(self, title):
        print(title, title in self)
        return self.get(title, None)

    def get_node_neighbors(self, node):
        for link in node.get_neighbors():
            yield self.get_node(link)
