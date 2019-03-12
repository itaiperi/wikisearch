from wikisearch.astar_elements import AstarSetElement, AstarSet
from .strategy import Strategy


class DefaultAstarStrategy(Strategy):
    """
    The A* classic strategy
    """

    def get_next_state(self, open_set: AstarSet) -> AstarSetElement:
        return open_set.get_min_f()
