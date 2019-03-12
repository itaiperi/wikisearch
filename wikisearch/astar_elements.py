from sortedcontainers import SortedList


class AstarSetElement(object):
    """
        Represents an element in an A*'s set
    """

    def __init__(self, state, f, g):
        self._state = state
        self._f = f
        self._g = g

    @property
    def state(self):
        return self._state

    @property
    def f(self):
        return self._f

    @property
    def g(self):
        return self._g

    def __eq__(self, other):
        return self.f == other.f and self.g == other.g and self.state == other.state

    def __lt__(self, other):
        return self.f < other.f or (self.f == other.f and self.state.title < other.state.title)

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other


class AstarSet(object):
    """
        Represents a set in the A* algorithm
    """

    def __init__(self):
        self._sorted_list = SortedList()
        self._dict = dict()

    def __delitem__(self, state):
        element = self._dict.pop(state)
        self._sorted_list.remove(element)

    def __getitem__(self, state):
        return self._dict[state]

    def __setitem__(self, state, f_g_tuple):
        element = AstarSetElement(state, f=f_g_tuple[0], g=f_g_tuple[1])
        old_element = self._dict.pop(state, None)
        if old_element:
            self._sorted_list.remove(old_element)
        self._dict[state] = element
        self._sorted_list.add(element)

    def __contains__(self, state):
        return state in self._dict

    def get_min_f(self):
        return self._sorted_list[0]

    def __len__(self):
        return len(self._sorted_list)
