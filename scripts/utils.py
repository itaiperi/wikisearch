import inspect
import os
import pickle


def print_progress_bar(iteration, total, elapsed_time=None, prefix='', suffix='', decimals=1, length=100, fill='█'):
    # Taken from Stack Overflow: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} Progress |{bar}| {percent}% {iteration}/{total} {suffix} Complete.'
          + f' Elapsed time: {elapsed_time:.1f} seconds' if elapsed_time else '', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class Cache:
    _should_cache = not os.environ.get('WIKISEARCH_CACHE') in [None, 'False']
    cache_path = os.environ.get('WIKISEARCH_CACHE_PATH')
    if _should_cache and (cache_path is None or not os.path.exists(cache_path)):
        raise ValueError(
            'Caching requested, but caching path is incorrect. Please define WIKISEARCH_CACHE env variable.')
    print(
        f'Internal caching mechanism {"used." if _should_cache else "not used. To use it, set WIKISEARCH_CACHE (bool) and WIKISEARCH_CACHE_PATH (str) env variables."}')

    def __init__(self):
        self.cache_path = os.environ.get('WIKISEARCH_CACHE_PATH')
        # Wether to use cache or not

    def __getitem__(self, key):
        if not self._should_cache:
            return None
        caller_path = os.path.splitext(os.path.abspath(inspect.stack()[1][1]).replace('/', '_'))[0]
        key_path = os.path.join(self.cache_path, f'{caller_path}-{str(key)}.pkl')
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                return pickle.load(f)
        else:
            return None

    def __setitem__(self, key, value):
        if not self._should_cache:
            return None
        caller_path = os.path.splitext(os.path.abspath(inspect.stack()[1][1]).replace('/', '_'))[0]
        key_path = os.path.join(self.cache_path, f'{caller_path}-{str(key)}.pkl')
        with open(key_path, 'wb') as f:
            pickle.dump(value, f)
