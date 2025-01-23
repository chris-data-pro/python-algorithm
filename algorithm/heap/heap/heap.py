import heapq

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
            heapq.heappush(self.max_heap, -x)
        else:
            heapq.heappush(self.min_heap, x)

        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))


    def getMedian(self) -> float:
        if not self.max_heap:
            raise ValueError("The max_heap list is empty")

        if len(self.max_heap) > len(self.min_heap):
            return float(-self.max_heap[0])
        
        return (-self.max_heap[0] + self.min_heap[0]) / 2.0


