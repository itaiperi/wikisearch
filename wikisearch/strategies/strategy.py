from abc import ABCMeta, abstractmethod


class Strategy(metaclass=ABCMeta):
    """
    Represents the strategy by which the next state in the searching path will be chosen
    """

    @abstractmethod
    def get_next_state(self, open_set):
        """
        Returns the next state
        :param open_set: The states which are still relevant to walk by
        """
        raise NotImplementedError
