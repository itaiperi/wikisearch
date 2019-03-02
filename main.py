import argparse
import time

from scripts.loaders import load_embedder_from_model_path, load_model_from_path
from wikisearch.astar import Astar
from wikisearch.costs.uniform_cost import UniformCost
from wikisearch.graph import WikiGraph
from wikisearch.heuristics import BFSHeuristic
from wikisearch.heuristics.nn_heuristic import NNHeuristic
from wikisearch.strategies import DefaultAstarStrategy
from wikisearch.utils.clean_data import tokenize_title

if __name__ == '__main__':
    """
    Finding a path between two wikipedia pages    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='source', help="Source title")
    parser.add_argument(dest='dest', help="Destination title")
    parser.add_argument('-t', '--time_limit', type=float,
                        help="Time limit (seconds) for source-dest distance calculation")
    args = parser.parse_args()

    cost = UniformCost()
    # heuristic = BFSHeuristic()
    model = load_model_from_path('data/distance_datasets/600613_30000_3000_3000/fasttexttitle_embeddingsdistance2_asymmetricmseloss_sgd_0.001_256_1_3/model.pth')
    embedder = load_embedder_from_model_path('data/distance_datasets/600613_30000_3000_3000/fasttexttitle_embeddingsdistance2_asymmetricmseloss_sgd_0.001_256_1_3/model.pth')
    heuristic = NNHeuristic(model, embedder)
    strategy = DefaultAstarStrategy()
    graph = WikiGraph()
    astar = Astar(cost, heuristic, strategy, graph)

    start = time.time()
    path, distance, developed = \
        astar.run(tokenize_title(args.source), tokenize_title(args.dest), args.time_limit)
    if path:
        print(f"Path: {' -> '.join([node.title for node in path])}")
        print(f"Distance: {distance}")
    else:
        print(f"Path not found.")
    print(f"-TIME- Time taken: {time.time() - start:.1f}s, Number of nodes developed: {developed}, Number of heuristics calculated: {heuristic.count}")
