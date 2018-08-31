import argparse

from wikisearch.astar import Astar
from wikisearch.graph import WikiGraph
from wikisearch.heuristics import bfs_heuristic
from wikisearch.strategies.default_astar_strategy import DefaultAstarStrategy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True, help="Source title")
    parser.add_argument('-d', '--dest', required=True, help="Destination title")
    args = parser.parse_args()

    heuristic = bfs_heuristic
    strategy = DefaultAstarStrategy()
    graph = WikiGraph()
    astar = Astar(heuristic, strategy, graph)

    path, distance, developed = astar.run(args.source, args.dest)
    print("Path: ", end="")
    print(*[node.title for node in path], sep=" -> ")
    print("Distance:", distance)
    print("Number of nodes developed:", developed)
