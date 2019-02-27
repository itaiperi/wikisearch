import argparse
import cProfile
import pstats

from scripts.loaders import load_embedder_from_model_path
from scripts.loaders import load_model_from_path
from wikisearch.astar import Astar
from wikisearch.costs.uniform_cost import UniformCost
from wikisearch.graph import WikiGraph
from wikisearch.heuristics.nn_heuristic import NNHeuristic
from wikisearch.strategies import DefaultAstarStrategy


def test_astar_time():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to the model file. When running from linux - '
                                              'notice to not put a \'/\' after the file name')
    args = parser.parse_args()

    embedder = load_embedder_from_model_path(args.model)
    model = load_model_from_path(args.model)

    cost = UniformCost()
    strategy = DefaultAstarStrategy()
    graph = WikiGraph()
    astar = Astar(cost, NNHeuristic(model, embedder), strategy, graph)

    path, distance, developed = astar.run("Joe Biden", "Gulf War")
    print(f"Path: {astar.stringify_path(path)}, Distance: {distance}, # Developed: {developed}")


if __name__ == "__main__":
    cProfile.run('test_astar_time()', 'profiling_astar')
    p = pstats.Stats('profiling_astar')
    p.sort_stats('tottime').print_stats()
