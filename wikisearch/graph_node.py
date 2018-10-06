class GraphNode:
    def __init__(self, title, pid, text, links):
        self._title = title
        self._pid = pid
        self._text = text
        self._links = links

    @property
    def title(self):
        return self._title

    @property
    def pid(self):
        return self._pid

    def get_neighbors(self):
        for neighbor in self._links:
            yield neighbor

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text

    def __hash__(self):
        return hash(self._title)
