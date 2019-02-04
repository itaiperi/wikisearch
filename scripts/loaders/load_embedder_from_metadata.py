import json
from importlib import import_module
from os import path

from wikisearch.embeddings import EMBEDDINGS_MODULES


def load_embedder_from_model_path(model_location_path):
    model_dir_path = path.dirname(model_location_path)
    model_file_name = path.splitext(path.basename(model_location_path))[0]

    # Loads dynamically the relevant embedding class
    with open(path.join(model_dir_path, f"{model_file_name}.meta")) as f:
        model_metadata = json.load(f)
    embedding = model_metadata['embedder']['type']
    return load_embedder(embedding)


def load_embedder(embedder_type):
    embedding_module = import_module('.'.join(['wikisearch', 'embeddings', EMBEDDINGS_MODULES[embedder_type]]),
                                     package='wikisearch')
    embedding_class = getattr(embedding_module, embedder_type)

    return embedding_class()
