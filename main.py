import argparse
import time

from scripts.consts.model import MODEL_TYPE, NN_MODEL, FUNC_MODEL
from scripts.loaders import load_embedder_from_model_path, load_model_from_path, load_embedder_by_name, \
    load_distance_method
from wikisearch.astar import Astar
from wikisearch.costs.uniform_cost import UniformCost
from wikisearch.graph import WikiGraph
from wikisearch.heuristics import BFSHeuristic
from wikisearch.heuristics.nn_heuristic import NNHeuristic
from wikisearch.strategies import DefaultAstarStrategy

if __name__ == '__main__':
    """
    Finding a path between two wikipedia pages    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='source', help="Source title")
    parser.add_argument(dest='dest', help="Destination title")
    parser.add_argument('-c', '--cost', default=1, help='The cost price')
    parser.add_argument('-t', '--time_limit', type=float,
                        help="Time limit (seconds) for source-dest distance calculation")
    subparsers = parser.add_subparsers(help='sub-command help', dest=MODEL_TYPE)

    # Creates the parser for a nn model
    nn_parser = subparsers.add_parser(NN_MODEL, help='nn_model help')
    nn_parser.add_argument('-m', '--model', required=True, help='Path to the model file. When running from linux '
                                                                '- notice to not put a \'/\' after the file name')

    # Creates the parser for a distance heuristic model
    dist_h_parser = subparsers.add_parser(FUNC_MODEL, help='func_model help')
    dist_h_parser.add_argument('-dh', '--distance-heuristic', required=True, help='The heuristic distance method')
    dist_h_parser.add_argument('-e', '--embedding', help='The embedder name')

    args = parser.parse_args()

    cost = UniformCost(int(args.cost))
    strategy = DefaultAstarStrategy()
    graph = WikiGraph()

    if MODEL_TYPE == NN_MODEL:
        embedder = load_embedder_from_model_path(args.model)
        model = load_model_from_path(args.model)
        heuristic = NNHeuristic(model, embedder)
    if args.distance_heuristic == "BFSHeuristic":
        heuristic = BFSHeuristic()
    else:
        embedder = load_embedder_by_name(args.embedding)
        heuristic = load_distance_method(args.distance_heuristic, embedder)

    astar = Astar(cost, heuristic, strategy, graph)

    start = time.time()
    path, distance, developed = astar.run(args.source, args.dest, args.time_limit)
    if path:
        print(f"Path: {' -> '.join([node.title for node in path])}")
        print(f"Distance: {distance}")
    else:
        print(f"Path not found.")
    print(f"-TIME- Time taken: {time.time() - start:.1f}s, Number of nodes developed: {developed}, Number of heuristics calculated: {heuristic.count}")
