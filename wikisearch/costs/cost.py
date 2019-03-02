from abc import ABCMeta, abstractmethod


class Cost(metaclass=ABCMeta):
    """
    Represents the cost to progress to the next state
    """

    @abstractmethod
    def _calculate(self, curr_state, succ_state):
        """
        Calculates the cost to progress from the current state to the successor state
        :param curr_state: The current state
        :param succ_state: The successor state
        :return: cost between current and successor state
        """
        raise NotImplementedError

    def calculate(self, curr_state, succ_state):
        """
        Wrapper method for _calculate, to insert behavior that applies to all Cost classes
        :param curr_state: The current state
        :param succ_state: The successor state
        :return: cost between current and successor state
        """

        return self._calculate(curr_state, succ_state)
