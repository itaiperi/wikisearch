from wikisearch.heuristics.heuristic import Heuristic


class CosineDistance(Heuristic):
    def __init__(self, embedder, eps=1e-8):
        super(CosineDistance, self).__init__()
        self._embedder = embedder
        self.eps = eps

    def _calculate(self, curr_state, dest_state):
        curr_embed = self._embedder.embed(curr_state.title)
        dest_embed = self._embedder.embed(dest_state.title)
        # We calculate 1 / cosine-similarity, because we want distance, which is the opposite of similarity
        return (curr_embed.norm() * dest_embed.norm()).item() / max(curr_embed.matmul(dest_embed).item(), self.eps)
