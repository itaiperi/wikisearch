from abc import ABCMeta, abstractmethod


class Heuristic(metaclass=ABCMeta):
    """
    The heuristic base class
    """

    def __init__(self):
        self._count = 0

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, count):
        self._count = count

    def calculate(self, curr_state, dest_state):
        self._count += 1
        return self._calculate(curr_state, dest_state)

    @abstractmethod
    def _calculate(self, curr_state, dest_state):
        """
        Calculates the heuristic distance between the given states
        :param curr_state: The current state
        :param dest_state: The destination state
        """
        raise NotImplementedError
