from .utils.consts import *

class WikiGraph(dict):
    def __init__(self, csv_filepath):
        super()
        with open(csv_filepath) as csv_file:
            for line in csv_file.readlines():
                title, pid, url, text, links = line.split(CSV_SEPARATOR)
                self[title] = {'pid': pid, 'url': url, 'text': text, 'links': links.split[LINKS_SEPARATOR]}

    def getNode(self, title):
        return self.get(title, None)
