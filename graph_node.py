class GraphNode:
    def __init__(self, title, pid, url, text, links):
        self.title = title
        self.pid = pid
        self.url = url
        self.text = text
        self.links = links

    def get_neighbors(self):
        return self.links

    def get_text(self):
        return self.text

    def __hash__(self):
        return hash(self.title)
