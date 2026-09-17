from collections import OrderedDict


class MessageCache:
    def __init__(self, max_size: int = 5000):
        '''cache consumes approximately 0.5 kilobytes for only one element'''
        self._cache = OrderedDict()
        self._max_size = max_size
 

    def set_if_absent(self, key:str, value: str):

        if key in self._cache:
            return False

        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = value
        return True



    def get(self, mid: str) -> str | None:
        if mid in self._cache:
            self._cache.move_to_end(mid)
            return self._cache[mid]
        return None