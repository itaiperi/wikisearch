from wikisearch.costs.cost import Cost


class UniformCost(Cost):
    def calculate(self, curr_state, next_state):
        return 1
