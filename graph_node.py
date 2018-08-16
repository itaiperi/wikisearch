from .utils.consts import *

class GraphNode(dict):
    def __init__(self, title, pid, url, text, links):
        self.title = title
        self.pid = pid
        self.url = url
        self.text = text
        self.links = links

    def getNeighbors(self):
        return self.links

    def getText(self):
        return self.text
