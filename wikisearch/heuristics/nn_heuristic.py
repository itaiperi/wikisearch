import torch

from .heuristic import Heuristic
from .nn_archs import EmbeddingsDistance


class EmbeddingsNNHeuristic(Heuristic):
    def __init__(self, model_state_path, embed_dim, embedder):
        self._model = EmbeddingsDistance(embed_dim)
        self._model.load_state_dict(torch.load(model_state_path))
        self._embedder = embedder

    def calculate(self, curr_state, dest_state):
        curr_embedding = self._embedder.embed(curr_state.title).unsqueeze(0)
        dest_embedding = self._embedder.embed(dest_state.title).unsqueeze(0)
        return self._model(curr_embedding, dest_embedding).item()
