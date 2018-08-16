from functools import reduce

from .strategy import Strategy


class DefaultAstarStrategy(Strategy):
    def get_next_state(self, open_set):
        min_f = float("inf")
        min_g = 0
        min_state = None
        for state, scores in open_set.items():
            if scores['f_score'] < min_f:
                min_f = scores['f_score']
                min_state = state
                min_g = scores['g_score']

        return min_state, min_g
