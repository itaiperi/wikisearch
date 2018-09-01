from abc import ABCMeta, abstractmethod


class Cost(metaclass=ABCMeta):
    @abstractmethod
    def calculate(self, curr_state, dest_state):
        raise NotImplementedError
