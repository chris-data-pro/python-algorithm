from collections import OrderedDict


class LRUCache:
    """
    134
    Implement a data structure for Least Recently Used (LRU) cache. It should support the following operations:
    get(key) Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
    set(key, value) Set or insert the value if the key is not already present. When the cache reached its capacity,
    it should invalidate the least recently used item before inserting a new item.

    Input:
    LRUCache(2)
    set(2, 1)
    set(1, 1)
    get(2)
    set(4, 1)
    get(1)
    get(2)
    Output: [1,-1,1]

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


class KeyNode(object):
    def __init__(self, key, value, freq = 1):
        self.key = key
        self.value = value
        self.freq = freq
        self.prev = self.next = None


class FreqNode(object):
    def __init__(self, freq, prev, next):
        self.freq = freq
        self.prev = prev
        self.next = next
        self.first = self.last = None


class LFUCache(object):
    """
    LFU (Least Frequently Used) is a famous cache eviction algorithm.
    For a cache with capacity k, if the cache is full and need to evict a key in it,
    the key with the least frequently used will be kicked out.

    Input:
    LFUCache(3)
    set(2,2)
    set(1,1)
    get(2)
    get(1)
    get(2)
    set(3,3)
    set(4,4)  # the cache with value 3 is not used, so it is replaced.
    get(3)
    get(2)
    get(1)
    get(4)

    Output:
    2
    1
    2
    -1
    2
    1
    4
    """

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.keyDict = dict()
        self.freqDict = dict()
        self.head = None

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.keyDict:
            keyNode = self.keyDict[key]
            value = keyNode.value
            self.increase(key, value)
            return value
        return -1

    def set(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if self.capacity == 0:
            return
        if key in self.keyDict:
            self.increase(key, value)
            return
        if len(self.keyDict) == self.capacity:
            self.removeKeyNode(self.head.last)
        self.insertKeyNode(key, value)

    def increase(self, key, value):
        """
        Increments the freq of an existing KeyNode<key, value> by 1.
        :type key: str
        :rtype: void
        """
        keyNode = self.keyDict[key]
        keyNode.value = value
        freqNode = self.freqDict[keyNode.freq]
        nextFreqNode = freqNode.next
        keyNode.freq += 1
        if nextFreqNode is None or nextFreqNode.freq > keyNode.freq:
            nextFreqNode = self.insertFreqNodeAfter(keyNode.freq, freqNode)
        self.unlinkKey(keyNode, freqNode)
        self.linkKey(keyNode, nextFreqNode)

    def insertKeyNode(self, key, value):
        """
        Inserts a new KeyNode<key, value> with freq 1.
        :type key: str
        :rtype: void
        """
        keyNode = self.keyDict[key] = KeyNode(key, value)
        freqNode = self.freqDict.get(1)
        if freqNode is None:
            freqNode = self.freqDict[1] = FreqNode(1, None, self.head)
            if self.head:
                self.head.prev = freqNode
            self.head = freqNode
        self.linkKey(keyNode, freqNode)

    def delFreqNode(self, freqNode):
        """
        Delete freqNode.
        :rtype: void
        """
        prev, next = freqNode.prev, freqNode.next
        if prev: prev.next = next
        if next: next.prev = prev
        if self.head == freqNode: self.head = next
        del self.freqDict[freqNode.freq]

    def insertFreqNodeAfter(self, freq, node):
        """
        Insert a new FreqNode(freq) after node.
        :rtype: FreqNode
        """
        newNode = FreqNode(freq, node, node.next)
        self.freqDict[freq] = newNode
        if node.next: node.next.prev = newNode
        node.next = newNode
        return newNode

    def removeKeyNode(self, keyNode):
        """
        Remove keyNode
        :rtype: void
        """
        self.unlinkKey(keyNode, self.freqDict[keyNode.freq])
        del self.keyDict[keyNode.key]

    def unlinkKey(self, keyNode, freqNode):
        """
        Unlink keyNode from freqNode
        :rtype: void
        """
        next, prev = keyNode.next, keyNode.prev
        if prev: prev.next = next
        if next: next.prev = prev
        if freqNode.first == keyNode: freqNode.first = next
        if freqNode.last == keyNode: freqNode.last = prev
        if freqNode.first is None: self.delFreqNode(freqNode)

    def linkKey(self, keyNode, freqNode):
        """
        Link keyNode to freqNode
        :rtype: void
        """
        firstKeyNode = freqNode.first
        keyNode.prev = None
        keyNode.next = firstKeyNode
        if firstKeyNode: firstKeyNode.prev = keyNode
        freqNode.first = keyNode
        if freqNode.last is None: freqNode.last = keyNode


"""
The read4 API is already defined for you.
@param buf a list of characters
@return an integer
you can call Reader.read4(buf)
"""


class ReadNByRead4:
    """
    660
    The API: int read4(char *buf) reads 4 characters at a time from a file.
    The return value is the actual number of characters read.
    For example, it returns 3 if there is only 3 characters left in the file.
    By using the read4 API, implement the function int read(char *buf, int n) that reads n characters from the file.
    The read function may be called multiple times.

    Input:
    "filetestbuffer"
    read(6)
    read(5)
    read(4)
    read(3)
    read(2)
    read(1)
    read(10)

    Output:
    6, buf = "filete"
    5, buf = "stbuf"
    3, buf = "fer"
    0, buf = ""
    0, buf = ""
    0, buf = ""
    0, buf = ""
    """

    def __init__(self):
        self.buf4, self.i4, self.n4 = [None] * 4, 0, 0

    # @param {char[]} buf destination buffer
    # @param {int} n maximum number of characters to read
    # @return {int} the number of characters read
    def read(self, buf, n):
        # Write your code here
        i = 0
        while i < n:
            if self.i4 == self.n4:
                self.i4, self.n4 = 0, Reader.read4(self.buf4)
                if not self.n4:
                    break
            buf[i], i, self.i4 = self.buf4[self.i4], i + 1, self.i4 + 1
        print(self.i4, self.n4, self.buf4)
        return i
