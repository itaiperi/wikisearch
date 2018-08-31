import argparse
import os

from wikisearch.astar import Astar
from wikisearch.graph import WikiGraph
from wikisearch.heuristics.bfs_heuristic import BFSHeuristic
from wikisearch.strategies.default_astar_strategy import DefaultAstarStrategy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True, help="Source title")
    parser.add_argument('-d', '--dest', required=True, help="Destination title")
    parser.add_argument('-t', '--time_limit', type=float, default=60,
                        help="Time limit (seconds) for source-dest distance calculation")
    args = parser.parse_args()
    wiki_lang = os.environ.get("WIKISEARCH_LANG") or "simplewiki"

    heuristic = BFSHeuristic()
    strategy = DefaultAstarStrategy()
    graph = WikiGraph(wiki_lang)
    astar = Astar(heuristic, strategy, graph)

    path, distance, developed = astar.run(args.source, args.dest, args.time_limit)
    if path:
        print("Path: ", end="")
        print(*[node.title for node in path], sep=" -> ")
        print("Distance:", distance)
        print("Number of nodes developed:", developed)
    else:
        print("Path not found. Number of nodes developed:", developed)
