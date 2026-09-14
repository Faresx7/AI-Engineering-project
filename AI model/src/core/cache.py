from collections import OrderedDict


class MessageCache:
    def __init__(self, max_size: int = 5000):
        '''cache consumes approximately 0.5 kilobytes for only one element'''
        self._cache = OrderedDict()
        self._max_size = max_size

    def has(self, mid: str) -> bool:
        '''check if the message is in cache or not'''
        return mid in self._cache

    def add(self, mid: str, text: str) -> None:
        '''add the new message to cache and ensure that
        updated messages have been moved to the end to 
        avoid deleting it when we need it
        '''
        self._cache[mid] = text
        self._cache.move_to_end(mid)

        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False) 

    def get(self, mid: str) -> str | None:
        if mid in self._cache:
            self._cache.move_to_end(mid)
            return self._cache[mid]
        return None