import time

from sortedcontainers import SortedList


class AstarSetElement(object):
    def __init__(self, state, f, g):
        self._state = state
        self._f = f
        self._g = g

    @property
    def state(self):
        return self._state

    @property
    def f(self):
        return self._f

    @property
    def g(self):
        return self._g

    def __eq__(self, other):
        return self.f == other.f and self.g == other.g and self.state == other.state

    def __lt__(self, other):
        return self.f < other.f and self.g < other.g and self.state.title < other.state.title

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other


class AstarSet(object):
    def __init__(self):
        self._sorted_list = SortedList()
        self._dict = dict()

    def __delitem__(self, state):
        element = self._dict.pop(state)
        self._sorted_list.remove(element)

    def __getitem__(self, state):
        return self._dict[state]

    def __setitem__(self, state, f_g_tuple):
        element = AstarSetElement(state, f=f_g_tuple[0], g=f_g_tuple[1])
        old_element = self._dict.pop(state, None)
        if old_element:
            print(old_element.state.title, old_element.f, old_element.g, old_element in self._sorted_list)
            for ele in self._sorted_list:
                if ele == old_element:
                    print(ele, old_element)
            self._sorted_list.remove(old_element)
        self._dict[state] = element
        self._sorted_list.add(element)

    def __contains__(self, state):
        return state in self._dict

    def get_min_f(self):
        return self._sorted_list[0]


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
        :return: (path, path_length, developed_nodes_amount) - The most successful path between the given titles
        and its length, the length of the path and how much nodes have been developed
        """
        source_state = self._graph.get_node(source_title)
        dest_state = self._graph.get_node(destination_title)

        parents = dict()
        closed_set = AstarSet()
        open_set = AstarSet()
        open_set[source_state] = (self._heuristic.calculate(source_state, dest_state), 0)

        developed = 0
        start_time = time.time()

        while open_set and ((time_limit is None) or (time.time() - start_time < time_limit)):
            next_state_element = self._strategy.get_next_state(open_set)
            next_state, next_g = next_state_element.state, next_state_element.g
            # f value doesn't matter in closed_set
            closed_set[next_state] = (0, next_g)
            del open_set[next_state]

            if next_state == dest_state:
                return self._reconstruct_path(parents, next_state), closed_set[next_state].g, developed

            developed += 1
            for succ_state in self._graph.get_node_neighbors(next_state):
                new_g = closed_set[next_state].g + self._cost.calculate(next_state, succ_state)
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
