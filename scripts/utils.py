import inspect
import os
import pickle


def print_progress_bar(iteration: int, total: int, elapsed_time: object = None, prefix: str = '', suffix: str = '',
                       decimals: int = 1, length: int = 100, fill: str = '█'):
    # Taken from Stack Overflow: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    Call in a loop to create terminal progress bar
    :param iteration: current iteration - Required
    :param total: total iterations - Required
    :param elapsed_time: Not in use
    :param prefix: prefix string - Optional
    :param suffix: suffix string - Optional
    :param decimals: positive number of decimals in percent complete - Optional
    :param length: character length of bar - Optional
    :param fill: bar fill character - Optional
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
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
        f'-INFO- Internal caching mechanism '
        f'{"used." if _should_cache else "not used. To use it, set WIKISEARCH_CACHE (bool) and WIKISEARCH_CACHE_PATH (str) env variables."}')

    def __getitem__(self, key):
        if not self._should_cache:
            return None
        caller_path = os.path.splitext(os.path.abspath(inspect.stack()[1][1]).replace(':', '').replace(os.sep, '_'))[0]
        key_path = os.path.join(self.cache_path, f'{caller_path}-{str(key)}.pkl')
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                return pickle.load(f)
        else:
            return None

    def __setitem__(self, key, value):
        if not self._should_cache:
            return None
        caller_path = os.path.splitext(os.path.abspath(inspect.stack()[1][1]).replace(':', '').replace(os.sep, '_'))[0]
        key_path = os.path.join(self.cache_path, f'{caller_path}-{str(key)}.pkl')
        with open(key_path, 'wb') as f:
            pickle.dump(value, f)
