from wikisearch.costs.cost import Cost


class UniformCost(Cost):
    """
    Represents the uniform cost. The distance from the current state to each of its successors is equal
    """
    def __init__(self, cost):
        super(UniformCost, self).__init__()
        self.cost = cost

    def _calculate(self, curr_state, next_state):
        return self.cost
