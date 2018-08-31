from abc import ABCMeta, abstractmethod


class Heuristic(metaclass=ABCMeta):
    @abstractmethod
    def calculate(self, curr_state, dest_state):
        raise NotImplementedError
