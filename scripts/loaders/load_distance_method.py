from importlib import import_module

from wikisearch.heuristics import HEURISTICS_DISTANCES_MODULES


def load_distance_method(distance_method, embedder):
    heuristic_module = \
        import_module('.'.join(['wikisearch', 'heuristics', HEURISTICS_DISTANCES_MODULES[distance_method]]),
                      package='wikisearch')
    heuristic_class = getattr(heuristic_module, distance_method)

    return heuristic_class(embedder)
