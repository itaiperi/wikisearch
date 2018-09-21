import itertools
from collections import defaultdict

from wikisearch.heuristics.heuristic import Heuristic


class BoWIntersection(Heuristic):
    def __init__(self, repeat=False):
        self._repeat = repeat

    def calculate(self, curr_state, dest_state):
        curr_state_text = curr_state.text
        dest_state_text = dest_state.text
        if self._repeat:
            # Add a suffix to each word, so that same words are mapped to different words, to count repetitions
            word_counter = defaultdict(lambda: itertools.count())
            curr_state_text = [word + '^%d' % next(word_counter[word]) for word in curr_state_text]
            word_counter = defaultdict(lambda: itertools.count())
            dest_state_text = [word + '^%d' % next(word_counter[word]) for word in dest_state_text]
        curr_words = set(curr_state_text)
        dest_words = set(dest_state_text)
        curr_words_size = len(curr_words)
        dest_words_size = len(dest_words)
        # Divide by size of destination state because that is the state that is important and we're trying to get to
        return len(curr_words & dest_words) / float(dest_words_size)
