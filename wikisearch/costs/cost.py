from abc import ABCMeta, abstractmethod


class Cost(metaclass=ABCMeta):
    """
    Represents the cost to progress to the next state
    """

    @abstractmethod
    def calculate(self, curr_state, succ_state):
        """
        Calculates the cost to progress from the current state to the successor state
        :param curr_state: The current state
        :param succ_state: The successor state
        """
        raise NotImplementedError
