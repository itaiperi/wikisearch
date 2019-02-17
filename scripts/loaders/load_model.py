import json
import os
from importlib import import_module

from wikisearch.consts.nn import EMBEDDING_VECTOR_SIZE
from wikisearch.heuristics.nn_archs import *


def load_model_from_path(model_location_path):
    metadata_path = os.path.splitext(model_location_path)[0] + ".meta"

    with open(metadata_path) as meta_file:
        metadata = json.load(meta_file)
    model = load_model_type(metadata['model']['arch_type'], metadata['model']['embed_dim'])
    model.load_state_dict(
        torch.load(model_location_path, map_location=None if torch.cuda.is_available() else 'cpu'))
    model.eval()
    return model


def load_model_type(model_type, embedding_size=EMBEDDING_VECTOR_SIZE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nn_arch_module = import_module('.'.join(['wikisearch', 'heuristics', 'nn_archs']), package='wikisearch')
    embedding_class = getattr(nn_arch_module, model_type)

    return embedding_class(embedding_size).to(device)
