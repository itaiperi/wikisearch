from wikisearch.astar import AstarSetElement
from .strategy import Strategy


class DefaultAstarStrategy(Strategy):
    """
    The A* classic strategy
    """

    def get_next_state(self, open_set) -> AstarSetElement:
        return open_set.get_min_f()
