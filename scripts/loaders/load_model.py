import torch

from wikisearch.consts.nn import EMBEDDING_VECTOR_SIZE
from wikisearch.heuristics.nn_archs import EmbeddingsDistance


def load_model(model_location_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingsDistance(EMBEDDING_VECTOR_SIZE).to(device)
    model.load_state_dict(
        torch.load(model_location_path, map_location=None if torch.cuda.is_available() else 'cpu'))
    model.eval()
    return model
