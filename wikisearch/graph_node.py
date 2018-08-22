class GraphNode:
    def __init__(self, title, pid, url, text, links):
        self._title = title
        self._pid = pid
        self._url = url
        self._text = text
        self._links = links

    def get_neighbors(self):
        return self._links

    def get_text(self):
        return self._text

    def __hash__(self):
        return hash(self._title)
