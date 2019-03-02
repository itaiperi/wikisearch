from wikisearch.heuristics.heuristic import Heuristic


class EuclideanDistance(Heuristic):
    def __init__(self, embedder,):
        super(EuclideanDistance, self).__init__()
        self._embedder = embedder
        self.p = 2

    def _calculate(self, curr_state, dest_state):
        curr_embed = self._embedder.embed(curr_state.title)
        dest_embed = self._embedder.embed(dest_state.title)
        return ((curr_embed - dest_embed) ** self.p).sum() ** (1 / self.p)
