from wikisearch.graph_node import GraphNode


class WikiGraph(dict):
    def __init__(self, csv_filepath):
        super()

        with open(csv_filepath) as csv_file:
            for line in csv_file.readlines():
                title, pid, url, text, links = line.split(CSV_SEPARATOR)
                self[title] = GraphNode(title, pid, url, text, links.split(LINKS_SEPARATOR))

    def get_node(self, title):
        return self.get(title, None)

    def get_node_neighbors(self, node):
        for link in node.get_neighbors():
            node = self.get_node(link)
            if node:
                yield self.get_node(link)
