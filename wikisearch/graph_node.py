class GraphNode:
    """
    A Node in the wikipedia graph. Each node represents a page in wikipedia
    """

    def __init__(self, title, pid, text, links, categories):
        """
        :param title: title of the entry, string
        :param pid: pageID, string
        :param text: text of the entry, string
        :param links: list of strings, which are titles of neighbors
        :param categories: list of strings, categories of the entry
        """
        self._title = title
        self._pid = pid
        self._text = text
        self._links = links
        self._categories = categories

    @property
    def title(self):
        """
        The node's title
        """
        return self._title

    @property
    def pid(self):
        """
        The node's pid
        """
        return self._pid

    @property
    def categories(self):
        return self._categories

    @property
    def neighbors(self):
        """
        The node's neighbors. Each existing link from the wikipedia page consider as a neighbor
        """
        for neighbor in self._links:
            yield neighbor

    @property
    def text(self):
        """
        The node's text
        """
        return self._text

    @text.setter
    def text(self, text):
        self._text = text

    def __hash__(self):
        return hash(self._title)
