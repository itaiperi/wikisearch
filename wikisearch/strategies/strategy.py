from abc import ABCMeta, abstractmethod


class Strategy(metaclass=ABCMeta):
    @abstractmethod
    def get_next_state(self, open_set):
        raise NotImplementedError
