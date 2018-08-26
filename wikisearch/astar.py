from collections import OrderedDict


class Astar():
    def __init__(self, heuristic, strategy, graph):
        self._heuristic = heuristic
        self._strategy = strategy
        self._graph = graph

    def run(self, source_title: str, dest_title: str) -> (list, int, int):
        source_state = self._graph.get_node(source_title)
        dest_state = self._graph.get_node(dest_title)

        closed_set = OrderedDict()
        parents = OrderedDict()
        open_set = OrderedDict([(source_state, {'f_score': self._heuristic(source_state.text, dest_state.text), 'g_score': 0})])

        developed = 0

        while open_set:
            next_state, g_score = self._strategy.get_next_state(open_set)
            closed_set[next_state] = g_score
            del open_set[next_state]

            if next_state == dest_state:
                return self._reconstruct_path(parents, next_state), closed_set[next_state], developed

            developed += 1
            for succ_state in self._graph.get_node_neighbors(next_state):
                new_g = closed_set[next_state] + 1
                if succ_state in open_set:
                    if new_g < open_set[succ_state]['g_score']:
                        parents[succ_state] = next_state
                        open_set[succ_state] = {'f_score': new_g + self._heuristic(succ_state.text, dest_state.text), 'g_score': new_g}
                else:
                    if succ_state in closed_set:
                        if new_g < closed_set[succ_state]:
                            parents[succ_state] = next_state
                            closed_set.pop(succ_state)
                            open_set[succ_state] = {'f_score': new_g + self._heuristic(succ_state.text, dest_state.text), 'g_score': new_g}
                    else:
                        parents[succ_state] = next_state
                        open_set[succ_state] = {'f_score': new_g + self._heuristic(succ_state.text, dest_state.text), 'g_score': new_g}

        # Reach here if there's no path between source and destination
        return None, -1, -1

    @staticmethod
    def _reconstruct_path(parents, dest_state):
        parents_list = []
        parent = dest_state

        while parent:
            parents_list = [parent] + parents_list
            parent = parents.get(parent, None)

        return parents_list
