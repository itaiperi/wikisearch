from abc import ABCMeta, abstractmethod


class Heuristic(metaclass=ABCMeta):
    """
    The heuristic base class
    """

    @abstractmethod
    def calculate(self, curr_state, dest_state):
        """
        Calculates the heuristic distance between the given states
        :param curr_state: The current state
        :param dest_state: The destination state
        """
        raise NotImplementedError
