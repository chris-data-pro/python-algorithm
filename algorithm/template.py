import unittest
from itertools import combinations, permutations
from collections import OrderedDict, deque
from heapq import heapreplace, heappushpop, heappush, heappop

'''
sorted(list(permutations([5, 1, 4])))  # list of tuples: [(1, 4, 5), (1, 5, 4), (4, 1, 5), (4, 5, 1), (5, 1, 4), (5, 4, 1)]
list(combinations([5, 1, 4], 2))  # list of tuples: [(5, 1), (5, 4), (1, 4)]
list(reversed([5, 1, 4]))  # [4, 1, 5]
'''

'''
cache = OrderedDict()
cache.move_to_end(key)
cache.popitem(last=False)  # this is FIFO, last=True -> LIFO
'''

'''
queue = deque()
queue.appendleft(el)
queue.popleft()
'''

'''
heap = []
heappush(heap, item)
heappop(heap)
heappushpop(heap, item)  # Push item first, then pop and return the smallest item from the heap. could pop the new
heapreplace(heap, item)  # Pop and return the smallest item from the heap, and then push the new item.
'''


class Solution:
    def my_func(self, input):

        return 10

    def say_hello(self, cost, t):
        if not cost:
            return

        i = 0
        # left = t
        left = cost[0] + (t - cost[0]) % sum(cost) if t > (cost[0] + sum(cost)) else t
        if left < cost[0]:
            return

        while left >= cost[i]:
            left -= cost[i]

            if i == len(cost) - 1:
                if left < cost[0]:
                    return i
                i = 0
            else:
                i += 1

        return i - 1


# class TestSolution(unittest.TestCase):
#     def setUp(self):
#         self.s = Solution()
#         self.input_1 = []
#         self.result_1 = 0
#         self.input_2 = []
#         self.result_2 = 0
#         self.input_3 = []
#         self.result_3 = 0
#
#     def test_my_func(self):
#         self.assertEqual(self.s.my_func(self.input_1), self.result_1)
#         self.assertEqual(self.s.my_func(self.input_2), self.result_2)
#         self.assertEqual(self.s.my_func(self.input_3), self.result_3)


if __name__ == '__main__':
    # unittest.main()
    s = Solution()
    # print(s.say_hello([5, 1, 4], 20))  # expect 0
    print(s.my_func(1))


'''

# A sender has a budget of t tokens, and needs to send packets to a list of n destinations. Sending to each destination 
# uses some number of tokens, stored in an array cost. Sending a packet to destination i uses cost[i] tokens.
# The sender rotates through each destination: after it sends to destination i, it sends to destination i + 1 
# (after destination n, send to destination 0).
# What is the last destination the sender will send to before it does not have enough tokens to send the next packet?
# Assume cost[i] > 0

# Example 1:
# Input: cost = [5,1,4], t = 22
# Output: 2
# Explanation: 
# - Send to destination 0 uses 5 tokens, so t = 17.
# - Send to destination 1 uses 1 token, so t = 16.
# - Send to destination 2 uses 4 tokens, so t = 12.
# - Send to destination 0 uses 5 tokens, so t = 7.
# - Send to destination 1 uses 1 token, so t = 6.
# - Send to destination 2 uses 4 tokens, so t = 2.
# There are not enough tokens to send to destination 0, so the last destination is destination 2.

'''







'''
Imagine a queue of 100 objects in a factory line.

o_0 o_1 o_2 o_3 o_4 ... o_98 o_99
^   ^   ^
|   |   |

In the front of the line, one out of the front three objects will be marked
uniformly at random. When this happens, the marked object and the object(s)
on the left of the marked object will be dequeued. The queue then moves forward
and this process repeats.

For example, if o_0 is marked, then o_0 will be dequeued and the new front
objects become o_1, o_2 and o_3. However, if o_1 is marked, o_0 and o_1 will be
dequeued, new front becomes o_2, o_3 and o_4.

For these 100 objects, which one has the highest chance to be marked?
'''

# o_0: 1 / 3
# o_1: 1 / 3 + 1 / 3 * (1 / 3)
# o_2: 1 / 3 + o_0 * (1 / 3 + (1 / 3) ** 2) + o_1 * (1 / 3) = 1/3 + 1/3 * o_0 + 1/3 * o_1
#
# o_n: 1 / 3 * o_(n - 3) + 1 / 3 * o_(n - 2) + 1 / 3 * o_(n - 1)

# queue = collections.deque([start])

'''
Welcome to Facebook!

This is just a simple shared plaintext pad, with no execution capabilities.

When you know what language you would like to use for your interview,
simply choose it from the dropdown in the top bar.

Enjoy your interview!
'''

'''
(1) Problem description:
        
A room is a m*n 2D matrix, we have the char array to represent the room.
In the array, X means a wall, O means free space, G means a guard:
You can only go with 4 directions: up, down, left and right.
    
Problem: Find the maximum distance between a space to the nearest guard.

    
XXXXOOOXXO
XXGOOXXOOG maximum distance is 5
GOOOXXGOXO 

    OOO   -> go left, left, down, left, left, to reach G.
  GOO    
    
    
For example, a space can reach to several guards, but we want the nearest guard.

XXX
XOX   the maximum distance is MaxValue/Infinity
XXX

OGO   the maximum distance is 1. 

OOGO  the maximum distance is 2. 

GOGO
OGOG  the maximum distance is 1. 

4 direction:
up
down
left
right

X: wall or blocks
O: free spaces
G: guard
    
    
    
OOO
OGO  max distance = 2
OOO

OOOOOOOOG max distance = 8
'''

'''
class Solution:
    
    def max_distance_o2g(self, input):
        if not input or not input[0]:
            return float('inf')
        
        rows, cols = len(input), len(input[0])
        up = left = 0
        down = rows - 1
        right = cols - 1
        
        self.visted = set()
        max_distance = 0
        
        i, j = 0, 0
        for i in range(rows):
            for j in range(cols):
                if input[i][j] == 'O':
                    self.visted.add((i, j))
                    max_distance = max(max_distance, self.helper(i, j, input, 1))
                    
        return max_distance
        
        
    def helper(self, x, y, matrix, maxd):
        if not matrix or not matrix[0]:
            return float('inf')
        
        count = maxd
        
        rows, cols = len(matrix), len(matrix[0])
        up = left = 0
        down = rows - 1
        right = cols - 1
        
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        if left <= x <= right and up <= y <= down:
            for dx, dy in directions:
                newx, newy = x + dx, y + dy
                
                if not (left <= newx <= right and up <= newy <= down):
                    continue
                
                if left <= newx <= right and up <= newy <= down and (newx, newx) not in self.visted:
                    if matrix[newx][newy] == 'O':
                        count += 1
                        self.helper(newx, newy, matrix, count)
                elif left <= newx <= right and up <= newy <= down and (newx, newx) not in self.visted:
                    if matrix[newx][newy] == 'G':
                        return maxd
                elif left <= newx <= right and up <= newy <= down and (newx, newx) not in self.visted:
                    if matrix[newx][newy] == 'X':
                        continue
                    
                self.visted.add((newx, newx))

        
        
O (M * N)

m no of cols
n no of rows
'''



class Solution:

    def max_distance_o2g(self, input):
        if not input or not input[0]:
            return float('inf')

        rows, cols = len(input), len(input[0])
        up = left = 0
        down = rows - 1
        right = cols - 1

        self.visted = set()
        max_distance = 0

        i, j = 0, 0
        for i in range(rows):
            for j in range(cols):
                if input[i][j] == 'O':
                    self.visted.add((i, j))
                    max_distance = max(max_distance, self.helper(i, j, input, 1))

        return max_distance

    def helper(self, x, y, matrix, maxd):
        if not matrix or not matrix[0]:
            return float('inf')

        count = maxd

        rows, cols = len(matrix), len(matrix[0])
        up = left = 0
        down = rows - 1
        right = cols - 1

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        if left <= x <= right and up <= y <= down:
            for dx, dy in directions:
                newx, newy = x + dx, y + dy

                if not (left <= newx <= right and up <= newy <= down):
                    continue

                if left <= newx <= right and up <= newy <= down and (newx, newx) not in self.visted:
                    if matrix[newx][newy] == 'O':
                        count += 1
                        self.helper(newx, newy, matrix, count)
                elif left <= newx <= right and up <= newy <= down and (newx, newx) not in self.visted:
                    if matrix[newx][newy] == 'G':
                        return maxd
                elif left <= newx <= right and up <= newy <= down and (newx, newx) not in self.visted:
                    if matrix[newx][newy] == 'X':
                        continue

                self.visted.add((newx, newx))


O(M * N)











