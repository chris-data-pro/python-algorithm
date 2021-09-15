from collections import OrderedDict


class LRUCache:
    """
    @param: capacity: An integer
    """

    def __init__(self, capacity):
        # do intialization if necessary
        self.capacity = capacity
        self.cache = OrderedDict()

    """
    @param: key: An integer
    @return: An integer
    """

    def get(self, key):
        # write your code here
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    """
    @param: key: An integer
    @param: value: An integer
    @return: nothing
    """

    def set(self, key, value):
        # write your code here
        if key not in self.cache:
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)  # this is FIFO, last=True -> FILO
        else:
            self.cache[key] = value
            self.cache.move_to_end(key)
