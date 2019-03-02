from wikisearch.heuristics.heuristic import Heuristic


class CosineSimilarity(Heuristic):
    def __init__(self, embedder, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self._embedder = embedder
        self.eps = eps

    def _calculate(self, curr_state, dest_state):
        curr_embed = self._embedder.embed(curr_state.title)
        dest_embed = self._embedder.embed(dest_state.title)
        return curr_embed.matmul(dest_embed).item() / max((curr_embed.norm() * dest_embed.norm()).item(), self.eps)
