import json
import os
import torch
from importlib import import_module


def load_model_from_path(model_location_path):
    metadata_path = os.path.splitext(model_location_path)[0] + ".meta"

    with open(metadata_path) as meta_file:
        metadata = json.load(meta_file)
    model = load_model_type(metadata['model']['arch_type'], metadata['model']['dims'])
    model.load_state_dict(
        torch.load(model_location_path, map_location=None if torch.cuda.is_available() else 'cpu'))
    model.eval()
    return model


def load_model_type(model_type, dims):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nn_arch_module = import_module('.'.join(['wikisearch', 'heuristics', 'nn_archs']), package='wikisearch')
    nn_arch_class = getattr(nn_arch_module, model_type)

    return nn_arch_class(dims).to(device)
