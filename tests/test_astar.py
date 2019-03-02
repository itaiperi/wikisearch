import argparse
import cProfile
import io
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
    pr = cProfile.Profile()
    pr.enable()
    path, distance, developed = astar.run("Joe Biden", "Gulf War")
    pr.disable()
    print(f"Path: {astar.stringify_path(path)}, Distance: {distance}, # Developed: {developed}")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()
    print(s.getvalue())


if __name__ == "__main__":
    test_astar_time()
