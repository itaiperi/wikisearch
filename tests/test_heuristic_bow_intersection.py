import unittest
from wikisearch.heuristics.bow_intersection import BoWIntersection
from wikisearch.graph_node import GraphNode


class TestBoWIntersection(unittest.TestCase):
    def setUp(self):
        self.curr_state = GraphNode(None, None, None, ["the", "dog", "barks", "Dog", "dog", "cat", "the"], None)
        self.dest_state = GraphNode(None, None, None, ["the", "cat", "meows", "the", "dog", "mouse"], None)

    def tearDown(self):
        del self.curr_state
        del self.dest_state

    def test_no_repeat(self):
        heuristic = BoWIntersection(repeat=False)
        self.assertEqual(heuristic.calculate(self.curr_state, self.dest_state), 3.0/5)

    def test_with_repeat(self):
        heuristic = BoWIntersection(repeat=True)
        self.assertEqual(heuristic.calculate(self.curr_state, self.dest_state), 4.0/6)


if __name__ == "__main__":
    unittest.main()
