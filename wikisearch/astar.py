import time

from wikisearch.astar_elements import AstarSet
from wikisearch.costs.cost import Cost
from wikisearch.graph import WikiGraph
from wikisearch.heuristics.heuristic import Heuristic
from wikisearch.strategies.strategy import Strategy


class Astar:
    """
    Implements the A* algorithm
    """

    def __init__(self, cost: Cost, heuristic: Heuristic, strategy: Strategy, graph: WikiGraph):
        self._cost = cost
        self._heuristic = heuristic
        self._strategy = strategy
        self._graph = graph

    def run(self, source_title: str, destination_title: str, time_limit: float = None) -> (list, int, int):
        """
        Runs A* from the source title to the destination title till gets the time limit
        :param source_title: The opening state of the path
        :param destination_title: The goal state of the path
        :param time_limit: The time assigned to the algorithm to run
        :return: (path, path_length, developed_nodes_amount) - The most successful path between the given titles
        and its length, the length of the path and how much nodes have been developed
        """
        source_state = self._graph.get_node(source_title)
        dest_state = self._graph.get_node(destination_title)

        self._heuristic.count = 0

        parents = dict()
        closed_set = AstarSet()
        open_set = AstarSet()
        open_set[source_state] = (self._heuristic.calculate(source_state, dest_state), 0)

        developed = 0
        start_time = time.time()

        while len(open_set) and ((time_limit is None) or (time.time() - start_time < time_limit)):
            next_state_element = self._strategy.get_next_state(open_set)
            next_state, next_g = next_state_element.state, next_state_element.g
            # f value doesn't matter in closed_set
            closed_set[next_state] = (0, next_g)
            del open_set[next_state]

            if next_state == dest_state:
                return self._reconstruct_path(parents, next_state), closed_set[next_state].g, developed

            developed += 1
            for succ_state in self._graph.get_node_neighbors(next_state):
                new_g = next_g + self._cost.calculate(next_state, succ_state)
                if succ_state in open_set:
                    if new_g < open_set[succ_state].g:
                        parents[succ_state] = next_state
                        open_set[succ_state] = (new_g + self._heuristic.calculate(succ_state, dest_state), new_g)
                else:
                    if succ_state in closed_set:
                        if new_g < closed_set[succ_state].g:
                            parents[succ_state] = next_state
                            del closed_set[succ_state]
                            open_set[succ_state] = (new_g + self._heuristic.calculate(succ_state, dest_state), new_g)
                    else:
                        parents[succ_state] = next_state
                        open_set[succ_state] = (new_g + self._heuristic.calculate(succ_state, dest_state), new_g)

        # Reach here if there's no path between source and destination
        return None, -1, developed

    @staticmethod
    def _reconstruct_path(parents, destination_state):
        """
        Reconstruct the path from the opening state till the destination state. The path is given backwards
        from the destination state through its parents till the opening state
        :param parents: the parents of each state from the destination state till the opening state
        :param destination_state: the destination state.
        :return: The reconstructed path from the opening state till the destination state
        """
        parents_list = []
        parent = destination_state

        while parent:
            parents_list = [parent] + parents_list
            parent = parents.get(parent, None)

        return parents_list

    @staticmethod
    def stringify_path(parents):
        return ' -> '.join([parent.title for parent in parents])
