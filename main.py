import argparse

from wikisearch.astar import Astar
from wikisearch.graph import WikiGraph
from wikisearch.heuristics import bfs_heuristic
from wikisearch.strategies.default_astar_strategy import DefaultAstarStrategy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--csv_file', required=True, help="Path to CSV file")
    parser.add_argument('-s', '--source', required=True, help="Source title")
    parser.add_argument('-d', '--dest', required=True, help="Destination title")
    args = parser.parse_args()

    heuristic = bfs_heuristic
    strategy = DefaultAstarStrategy()
    graph = WikiGraph(args.csv_file)
    astar = Astar(heuristic, strategy, graph)

    path, distance, developed = astar.run(args.source, args.dest)
    print(path)
    print(distance)
    print(developed)
