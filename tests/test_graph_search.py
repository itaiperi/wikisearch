import unittest

from wikisearch.astar import Astar
from wikisearch.consts.mongo import WIKI_LANG
from wikisearch.costs.uniform_cost import UniformCost
from wikisearch.graph import WikiGraph
from wikisearch.heuristics import BFSHeuristic
from wikisearch.strategies import DefaultAstarStrategy


class TestWikiGraphSearch(unittest.TestCase):
    def setUp(self):
        self.cost = UniformCost()
        self.heuristic = BFSHeuristic()
        self.strategy = DefaultAstarStrategy()
        self.graph = WikiGraph(WIKI_LANG)
        self.astar = Astar(self.cost, self.heuristic, self.strategy, self.graph)

    def tearDown(self):
        del self.cost, self.heuristic, self.strategy, self.graph, self.astar

    def test_follow_redirect(self):
        samples = [
            # Ralph Foody links to USA which redirects to United States
            ('Ralph Foody', 'United States', 1),
            # Ralph Foody -> United States -> Pacific Ocean
            ('Ralph Foody', 'Pacific Ocean', 2),
            # Midtown Manhattan redirects to Manhattan
            ('Midtown Manhattan', 'Manhattan', 0),
            ('New York City', 'Midtown Manhattan', 1),
            ('New York City', 'Manhattan', 1),
            # New York City -> Manhattan -> Stonewall riots
            ('New York City', 'Stonewall riots', 2),
        ]
        for source, destination, distance in samples:
            path_nodes, bfs_distance, _ = self.astar.run(source, destination)
            self.assertEqual(distance, bfs_distance, f'Path: {Astar.stringify_path(path_nodes)}')

    def test_shortest_path(self):
        samples = [
            # Chicken -> Chickenpox -> Herpes zoster #Shingles# -> Paresthesia #tingling# -> Alcoholism
            ('Chicken', 'Alcoholism', 4),
            # Inifinity (album) -> Journey (band) -> Heavy metal music
            ('Infinity (album)', 'Heavy metal music', 2),
        ]
        for source, destination, distance in samples:
            path_nodes, bfs_distance, _ = self.astar.run(source, destination)
            self.assertEqual(distance, bfs_distance, f'Path: {Astar.stringify_path(path_nodes)}')


if __name__ == "__main__":
    unittest.main()
