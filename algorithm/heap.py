from collections import defaultdict
from heapq import heappop, heappush


class MedianFinder:
    def __init__(self):
        self.left = []  # left half of the window, negative ordinal, e.g. [-99, -98, -97, ...]
        self.right = []  # right half of the window, ordinal e.g. [100, 100, 101, ..., 122]
        self.to_delete = defaultdict(int)  # only pop element when it gets to the top of the heap, O(1)
        self.left_size = 0
        self.right_size = 0

    def get_median(self):
        return chr(-self.left[0])

    def delete(self):
        while self.left and self.to_delete[-self.left[0]] > 0:
            self.to_delete[-self.left[0]] -= 1
            heappop(self.left)
        while self.right and self.to_delete[self.right[0]] > 0:
            self.to_delete[self.right[0]] -= 1
            heappop(self.right)

    def rebalance(self):
        while self.left_size > self.right_size + 1:
            heappush(self.right, -heappop(self.left))
            self.left_size -= 1
            self.right_size += 1
            self.delete()
        while self.left_size < self.right_size:
            heappush(self.left, -heappop(self.right))
            self.left_size += 1
            self.right_size -= 1
            self.delete()

    def add(self, c: str):
        if not self.left or ord(c) <= -self.left[0]:
            heappush(self.left, -ord(c))
            self.left_size += 1
        else:
            heappush(self.right, ord(c))
            self.right_size += 1
        self.rebalance()
    
    def remove(self, c: str):
        self.to_delete[ord(c)] += 1
        if ord(c) <= -self.left[0]:
            self.left_size -= 1
        else:
            self.right_size -= 1
        self.delete()
        self.rebalance()
    

def median_sliding_window(s, k):
    if not s or k <= 0:
        return []  # ''

    mf = MedianFinder()
    
    for i in range(k):
        mf.add(s[i])
    res = [mf.get_median()]
    
    for j in range(k, len(s)):
        mf.add(s[j])
        mf.remove(s[j - k])
        res.append(mf.get_median())

    return res  # ''.join(res)


# time complexity for the median_sliding_window is O(n log k)
# space complexity of the MedianFinder class is O(k)  (worst case O(n))

# s = input().strip()
# k = int(input())
# median_sliding_window(s, k)


if __name__ == '__main__':
    s = "aaaabbbb"
    print(median_sliding_window(s, 3))  # ['a', 'a', 'a', 'b', 'b', 'b']
    print(median_sliding_window(s, 4))  # ['a', 'a', 'a', 'b', 'b']
    
    # # given a character 'a', return the ordinal value of the character 97
    # ord('a')
    
    # # Given a the ordinal value of the character 97, return the character 'a'
    # chr(97)
    
    ss = "abcabc"
    print(median_sliding_window(ss, 3))  # ['b', 'b', 'b', 'b']
    print(median_sliding_window(ss, 4))  # ['a', 'b', 'b']


======================================================================================


"""
# Giving a list of integer coming in Radom order. Compute the median value of the list ?


#  List a =  {"2","1", "10", "8"}
# Object Solution {
#     def appendValue (x : Int) 
    
#     def getMedian() : Double 
# }
"""
class Solution:
    def __init__(self):
        self.values = []


    def appendValue(self, x):
        self.values.append(x)


    def getMedian(self) -> float: # O(NlogN) quick sort
        sorted_values = sorted(self.values)
        n = len(sorted_values)

        if n == 0:
            raise ValueError("The input list is empty")
        
        if n % 2 == 1:
            return sorted_values[n // 2]
        else:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0




class Solution:
    def __init__(self):
        self.max_heap = []
        self.min_heap = []


    def appendValue(self, x):
        if not self.max_heap or x <= -self.max_heap[0]:
            heappush(self.max_heap, -x)
        else:
            heappush(self.min_heap, x)

        if len(self.max_heap) > len(self.min_heap) + 1:
            heappush(self.min_heap, -heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heappush(self.max_heap, -heappop(self.min_heap))


    def getMedian(self) -> float:
        if not self.max_heap:
            raise ValueError("The max_heap list is empty")

        if len(self.max_heap) > len(self.min_heap):
            return float(-self.max_heap[0])
        
        return (-self.max_heap[0] + self.min_heap[0]) / 2.0
