from .bfs_heuristic import BFSHeuristic
from .bow_intersection import BoWIntersection
from .euclidean_distance import EuclideanDistance
from .cosine_distance import CosineDistance

HEURISTICS_DISTANCES_MODULES = {
    EuclideanDistance.__name__: 'euclidean_distance',
    CosineDistance.__name__: 'cosine_distance',
    BFSHeuristic.__name__: 'bfs_heuristic'
}
