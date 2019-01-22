import time
from collections import OrderedDict


class Astar:
    """
    Implements the A* algorithm
    """

    def __init__(self, cost, heuristic, strategy, graph):
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
        :return: The most successful path between the given titles and its length
        """
        source_state = self._graph.get_node(source_title)
        dest_state = self._graph.get_node(destination_title)

        closed_set = OrderedDict()
        parents = OrderedDict()
        open_set = OrderedDict([(source_state, {'f_score': self._heuristic.calculate(source_state.text, dest_state.text), 'g_score': 0})])

        developed = 0
        start_time = time.time()

        while open_set and ((time_limit is None) or (time.time() - start_time < time_limit)):
            next_state, g_score = self._strategy.get_next_state(open_set)
            closed_set[next_state] = g_score
            del open_set[next_state]

            if next_state == dest_state:
                return self._reconstruct_path(parents, next_state), closed_set[next_state], developed

            developed += 1
            for succ_state in self._graph.get_node_neighbors(next_state):
                new_g = closed_set[next_state] + self._cost.calculate(next_state, succ_state)
                if succ_state in open_set:
                    if new_g < open_set[succ_state]['g_score']:
                        parents[succ_state] = next_state
                        open_set[succ_state] = {'f_score': new_g + self._heuristic.calculate(succ_state.text, dest_state.text), 'g_score': new_g}
                else:
                    if succ_state in closed_set:
                        if new_g < closed_set[succ_state]:
                            parents[succ_state] = next_state
                            closed_set.pop(succ_state)
                            open_set[succ_state] = {'f_score': new_g + self._heuristic.calculate(succ_state.text, dest_state.text), 'g_score': new_g}
                    else:
                        parents[succ_state] = next_state
                        open_set[succ_state] = {'f_score': new_g + self._heuristic.calculate(succ_state.text, dest_state.text), 'g_score': new_g}

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
