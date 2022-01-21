"""
1. ask clarifying questions - gather requirements (scale, performance, APIs), documenting requirements right away
   scale: how many DAU users? transactions per second? Queries per second (QPS) API supports 500 transactions/second?
   functionalities: what functionalities do I need to implement?

2. translate requirements into application / system design

3. start with requirement analysis

4. propose meaningful functionalities / services

5. UI front end (button invokes an API) -
   web server, APIs (how the data is gonna be served) representation / services (payment, order) -
   database schema data model (entities, key attributes, relationships, security)

6. API: Domain/resource/{parameters} -
   Rest API uses HTTP requests to GET, POST, PUT (update) and DELETE data.
   HTTP: a request-response protocol between a client and server.
   http request POST to insert，send data (JSON object) to server (RDS) to create/update
   http request GET to request data from the server (RDS)
   Example: A client (browser) sends an HTTP request to the server; then the server returns a response to the client.

7. architecture - tradeoff

8. audience / users - security (internal / external) - reliability (consistency of response) -
   recovery (crash in-between and restart from last) - monitoring / logging (at different APIs)

9. ask for feedback - discuss options, pros and cons

10. focus on customer satisfaction and user experience
"""


"""
Data Schema:

I tend to lean towards a Relational Databases for most of this type of problems. Reasonably speaking you could probably 
go into more of a no sql types of solution. But just more of my expertise lies in the relational DB side. So presumably 
like some type of PostgresSQL, or MySQL is what we'll be working with here. So I'll be designing the data schema with 
that sort of perspective in mind.

id, PK, primary key, serial that way it can increment as you add more and more. that's totally reasonable
FK, presumably we're going to make a table down the line with this column. so we call it foreign key...
I think it's pretty reasonable to add ...

cool that seems pretty reasonable for a xxx table. Obviously we could end up adding, like user_id or such kind of fields
but like for now this is the important part. I think this is good to start with

?
For now I'm gonna call this a enum. There's sort of a TRADE_OFF there, when you end up adding enums. Particularly in 
like a Relational DB, just simply because it's more performant since you have all the benefits sort of being int
but the trade off is there's some flexibility issues down the line. I know in PostgreSQL specifically when you add enum
you can never remove it, like you can never remove the type afterwards. So that's like sth. to consider when you end up
choosing enum. I think for this specific case, like (compact, regular and large) I don't think those are things that 
like gonna be totally out of phase in a (parking garage) type of system. so I think it's totally reasonable to have enum 
here. (But like I mentioned before I don't think we're gonna run into issues of scale here, it'll be totally reasonable 
to be like just varchar as well, for like flexibility and then sort of have that vetting on the application layer side)

So yeah, I think that probably covers our requirements here. I mean obviously I can add more as we go through the 
problem but for now does that seem like a reasonable data model?
"""

"""
Next I wanna very briefly talk about The end to end data lifecycle. Obviously we already created the schema

Ingest: Actually the 1st stage in the data lifecycle, is ingest, or to pull in the raw data. such as streaming data 
        from devices, on-premises batch data, app logs, or mobile-app user events. Here in this particular example,
        I think it's reasonable to assume that ...
        
        (There are a number of ways to collect raw data, based on the data’s size, source, and latency)
        1. App: Data from app events, such as log files or user events, is typically collected in a PUSH model, where 
                the app calls an API to send the data to storage. 
                I know, When you host your apps on Google Cloud, you gain access to built-in tools to send your 
                data to Google Cloud’s ecosystem of data management services.
                for example: Writing data to a database: our app writes data to one of the databases that Google Cloud 
                provides, such as the managed MySQL of Cloud SQL (or the NoSQL databases provided by Datastore and Cloud 
                Bigtable.)
        
        ?        
        2. Streaming: The data consists of a continuous stream of small, asynchronous messages.
        3. Batch: Large amounts of data are stored in a set of files that are transferred to storage in bulk.
                  Bulk data consists of large datasets where ingestion requires high aggregate bandwidth between a small 
                  number of sources and the target. The data could be stored in files, such as CSV, JSON, or 
                  Parquet files, or in other relational or NoSQL database. 
                  The source data could be located on-premises or on other cloud platforms.
                  (For example: Moving data stored in an on-premises Oracle database to a fully managed Cloud SQL 
                               database using Informatica.)

Store: After the data has been retrieved, it needs to be stored in a format that is durable and can be easily accessed.
       We're talking about Storage like RDBMS, that you can use to store your relational data, 
       or NoSQL for non-relational data.
       
       For example, Cloud SQL is appropriate for typical (online transaction processing) OLTP workloads. such as
       1. Financial transactions: Storing financial transactions requires ACID database semantics, and data is often 
                                  spread across multiple tables, so complex transaction support is required.
       2. User credentials: Storing passwords or other secure data requires complex field support and enforcement along 
                            with schema validation.
       3. Customer orders: Orders or invoices typically include highly normalized relational data and multi-table 
                           transaction support when capturing inventory changes.

       ?
       Cloud SQL is not an appropriate storage system for (online analytical processing) OLAP workloads or data that 
       requires dynamic schemas on a per-object basis. (If your workload requires dynamic schemas, consider Datastore. 
       For OLAP workloads, consider BigQuery. If your workload requires wide-column schemas, consider Bigtable.)

Process and analyze: In this stage, the data is transformed from raw form into actionable information. Here presumably
                     we already have the data processed into the tables above. And then the analyze or querying, which I 
                     guess we will cover by actually doing it next.
                     
Explore and visualize: The final stage is to convert the results of the analysis into a format that is easy to draw 
                       INSIGHTS from and to share with colleagues and peers. This is using the SQL output results for 
                       some data science/visualization work. Hopefully the SQL I write next will be good enough for 
                       these purposes.
                    
Ok that's all I wanna cover for end to end data lifecycle. Let's move on ...
"""

import random
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


class SolutionCW:
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


# if __name__ == '__main__':
#     # unittest.main()
#     s = Solution()
#     # print(s.say_hello([5, 1, 4], 20))  # expect 0
#     print(s.my_func(1))


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


class FBSolution:

    def max_distance_o2g(self, input):
        if not input or not input[0]:
            return float('inf')

        rows, cols = len(input), len(input[0])
        up = left = 0
        down = rows - 1
        right = cols - 1

        max_distance = 0

        for i in range(rows):
            for j in range(cols):
                if input[i][j] == 'O':
                    self.visited = set([(i, j)])
                    self.res = rows * cols
                    self.helper(i, j, input, 0)
                    max_distance = max(max_distance, float('inf') if self.res == rows * cols else self.res)

        return max_distance

    def helper(self, x, y, matrix, steps):

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in directions:
            newx, newy = x + dx, y + dy

            if 0 <= newx < len(matrix) and 0 <= newy < len(matrix[0]) and (newx, newy) not in self.visited:
                if matrix[newx][newy] == 'G':
                    steps += 1
                    self.res = min(self.res, steps)
                    return
                elif matrix[newx][newy] == 'X':
                    continue
                elif matrix[newx][newy] == 'O':
                    steps += 1
                    self.visited.add((newx, newy))
                    self.helper(newx, newy, matrix, steps)
                    self.visited.remove((newx, newy))
                    steps -= 1
        return


class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
        self.left, self.right = None, None


class SolutionG:

    def make_horizontal_order(self, node):
        if not node:
            return []

        res = {0: [node]}
        level = 1

        while res[level - 1]:
            res[level] = []
            for i in res[level - 1]:
                if i.left:
                    res[level].append(i.left)
                if i.right:
                    res[level].append(i.right)

            level += 1

        return [[x for x in y] for y in res.values() if y]   # [[node(1)], [node(2), node(3)], ....]


    def make_ll(self, inputList):
        if not inputList:
            return []

        ans = [inputList[0][0]]  # 第一个错 不是[]
        for nodes in inputList:
            for i in range(len(nodes) - 1):
                if i == 0:  # 第二个错, 不是 i = 0
                    ans.append(nodes[i])
                nodes[i].next = nodes[i+1]

        return ans


    def make_horizontal_ll(self, node):
        if not node:
            return []

        res = {0: [node]}
        level = 1
        ans = [node]  # 第一个错 不是[[node]]

        while res[level - 1]:
            res[level] = []

            for i in res[level - 1]:

                if i.left:
                    if len(res[level]) == 0:
                        res[level].append(i.left)
                    else:
                        res[level].append(i.left)  # 第二个错
                        res[level][-1].next = i.left

                    if len(res[level]) == 1:
                        ans.append(res[level][0])

                if i.right:
                    if len(res[level]) == 0:
                        res[level].append(i.right)
                    else:
                        res[level].append(i.right)  # 第二个错
                        res[level][-1].next = i.right

                    if len(res[level]) == 1:
                        ans.append(res[level][0])

            level += 1

        return ans  # [head1, head2]


if __name__ == '__main__':
    fbs = FBSolution()
    print(fbs.max_distance_o2g([['X', 'X', 'X', 'X', 'O', 'O', 'O', 'X', 'X', 'O'],
                                ['X', 'X', 'G', 'O', 'O', 'X', 'X', 'O', 'O', 'G'],
                                ['G', 'O', 'O', 'O', 'X', 'X', 'G', 'O', 'X', 'O']]))  # expect 5
    print(fbs.max_distance_o2g([['X', 'X', 'X', 'X'],
                                ['O', 'O', 'G', 'O'],
                                ['X', 'X', 'X', 'X']]))  # expect 2
    print(fbs.max_distance_o2g([['X', 'X', 'X', 'X'],
                                ['X', 'O', 'O', 'X'],
                                ['X', 'X', 'X', 'X']]))  # expect inf
    print(fbs.max_distance_o2g([['O', 'O', 'O'],
                                ['O', 'G', 'O'],
                                ['O', 'O', 'O']]))  # expect 2

    # root
    #                  10
    #                 /  \
    #               -5   15
    #                    / \
    #                   6  20
    gs = SolutionG()
    root = Node(10)
    root.left = Node(-5)
    node_2 = Node(15)
    root.right = node_2
    node_2.left = Node(6)
    node_2.right = Node(20)
    il = gs.make_horizontal_order(root)
    print(il)
    print([[x.val, x.next.val if x.next else None] for x in gs.make_ll(il)])
    res = gs.make_horizontal_ll(root)
    print([[x.val, x.next.val if x.next else None] for x in res])






"""

Given an integer array and an integer number k. Return the k-th largest element in the array.

Examples:
array = [5, -3, 9, 1]
* k = 1 => return: 9
* k = 2 => return: 5
* k = 4 => return: -3



"""


# class Soluition:
#
#     def quick_select(self, L, start, end, k):  # k 0-indexed
#         if start == end:
#             return start, L[start]
#
#         left, right = start, end
#         pivot = L[(start + end) // 2]
#
#         while left <= right:
#             while left <= right and L[left] < pivot:
#                 left += 1
#             while left <= right and L[right] > pivot:
#                 right -=1
#
#             if left <= right:
#                 L[left], L[right] =  L[right], L[left]
#                 left += 1
#                 right -= 1
#
#         if start <= right and k <= right:
#             return self.quick_select(L, start, right, k)
#         if left <= end and k >= left:
#             return self.quick_select(L, left, end, k)
#
#         return L[k]
#
#
#     def kth_largest(self, array, k):
#         if not array or k <= 0:
#             return
#
#         res = self.quick_select(array, 0, len(array) - 1, len(array) - k)
#         return res


"""
array = [5, -3, 9, 1], k = 2
"""
# k = 2, left 0, right 3
# right 2
# right 1
#
# [-3, 5, 9, 1] left 1 right 0

#     def kth_largest_sorted(self, array, k):
#         if not L or k <= 0:
#             return

#         array.sort()


# O(NlogN)
#
# heapq
# O(N + (N - k)logN)




"""

Given a list of city names and their corresponding populations, write a function to output a random city name subject to the following constraint: the probability of the function to output a city's name is based on its population divided by the sum of all cities' population.
For example:

NY: 7
SF: 5
LA: 8
The probability to generate NY is 7/20, SF is 1/4.


"""

# input = {'NY': 7, 'SF': 5, 'LA': 8}
# input1 = []
# lv = []
#
# def random_city_name(input):
#     total = sum(input.values())
#     r = random.random()
#
#     lv = list(input.values())  # [5, 5, 8]
#     sum = 0
#     moving_sum = []
#     for v in lv:
#         sum += v
#         moving_sum.append(sum)  # [7, 12, 20]
#
#     for i in range(len(moving_sum) - 1):  # 0 to len(moving_sum) - 2
#         if r * total < moving_sum[i]:
#             return [item[0] for item in input.items() if item[1] == lv[i]][0]

        # ith key-value from a dict

# What if r * total = 6

# input = {'NY': 5, 'SF': 5, 'LA': 8}
# [5, 10, 18]
# r * total = 9


# [E] Given a binary search tree, design an algorithm which creates a linked list of all the nodes at each level. If the height of a binary search tree is k, we should have k linked lists.
# 1
# / \
#     2        3
# /  \    / \
#     4   5  6   7
# / \
#     8
#
# 0= 1
# 1= 2->3
# 2= 4->5->6->7
# 3=8
#
# class LinkedListNode:
#     def __init__(self, val, next=None):
#         self.val = val
#         self.next = next
#
# class Solution:
#
#     def make_horizontal_order(self, node):
#         if not node:
#             return []
#
#         res = {0: [node]}
#         level = 1
#
#         while res[level - 1]:
#             res[level] = []
#             for i in res[level - 1]:
#                 if i.left:
#                     res[level].append(i.left)
#                 if i.right:
#                     res[level].append(i.right)
#
#             level += 1
#
#         return [[x for x in y] for y in res.values() if y]   # [[node(1)], [node(2), node(3)], ....]
#
#
#     def make_ll(self, inputList):
#         if not inputList:
#             return []
#
#         ans = []
#         for nodes in inputList:
#             for i in range(len(nodes) - 1):
#                 if i = 0:
#                     ans.append(nodes[i])
#                 nodes[i].next = nodes[i+1]
#
#         return ans
#
#
#
# # O(n)
#
#
#
# class Solution2:
#
#     def make_horizontal_ll(self, node):
#         if not node:
#             return []
#
#         res = {0: [node]}
#         level = 1
#         ans = [[node]]
#
#         while res[level - 1]:
#             res[level] = []
#
#             for i in res[level - 1]:
#                 if i.left:
#                     if len(res[level]) = 0:
#                         res[level].append(i.left)
#                     else:
#                         res[level][-1].next = i.left
#
#                     if len(res[level]) = 1:
#                         ans.append(res[level][0])
#
#                 if i.right:
#                     if len(res[level]) = 0:
#                         res[level].append(i.right)
#                 else:
#                     res[level][-1].next = i.right
#
#                 if len(res[level]) = 1:
#                     ans.append(res[level][0])
#
#             level += 1
#
#         return ans  # [head1, head2]
#
#
#
# import unittest
#
# class Test
#
#
# 1
#
#
#
# [E] Stack with max api
#
# from heapq import heappush, heappop
#
# class MaxStack:
#     def __init__(self):
#         self.stack = []
#         self.count = 0
#         self.heap = []
#         self.popped_set = set()
#
#     def push(self, x):
#         item = (-x, -self.count)
#         self.stack.append(item)
#         heappush(self.heap, item)
#         self.count += 1
#
#     def _clear_popped_in_stack(self):
#         while self.stack and self.stack[-1] in self.popped_set:
#             self.popped_set.remove(self.stack[-1])
#             self.stack.pop()
#
#     def _clear_popped_in_heap(self):
#         while self.heap and self.heap[0] in self.popped_set:
#             self.popped_set.remove(self.heap[0])
#             self.stack.pop()
#
#
#     def popMax(self):
#         self._clear_popped_in_heap()
#         item = heappop(self.heap)
#         self.popped_set.add(item)
#         return -item[0]
#
#
#
#     def max_stack(self, inputs):
#         maxi = 0
#         for input in inputs:
#             if input > maxi:
#                 maxi = input
#             # maxi = max(maxi, input)
#
#         return maxi
#
#
#     # O(N)
#     # O(NlogN)


# SQL Question: given 2 tables parents and children, provide the sql to give me all parents that have at least one male child and one female child
#
# select pname
# from parents p
# where parent_id in
# (with mparents as (
#  select parent_id, count(distinct child_id) as mct
# from children
#     where child_gender = 'male'
# group by 1
# having count(distinct child_id) > 0
# ),
# fparents as (
#     select parent_id, count(distinct child_id) as fct
# from children
#     where child_gender = 'female'
# group by 1
# having count(distinct child_id) > 0
# )
# select l.parent_id
# from mparents l inner join fparents r
# on l.parent_id = r.parent_id
# order by 1)
# ;
#
# select pname
# from parents p
# where parent_id in
# (
#     select distinct parent_id
# from children
# where child_gender = 'male'
# )
# and parent_id in
# (
#     select parent_id
# from children
# where child_gender = 'female'
# )
# ;
#
#
# select pname
# from parents p
# join (
# select distinct parent_id
# from children
#     where child_gender = 'male'
# ) a
# on p.parent_id = a.parent_id
# join
# (
# select distinct parent_id
# from children
# where child_gender = 'female'
# ) b
# on p.parent_id = b.parent_id
# ;
#
# Python: Convert a sorted array into a binary search tree
# Input:  Array {1, 2, 3}
# Output: A Balanced BST
# 2
# / \
#     1    3
#
# Input: Array {4, 5, 6, 7}
# Output: A Balanced BST
# 6
# / \
#     5    7
# /
# 4
#
#
#
# 5
# / \
#     4    6
# \
# 7
#
# class TreeNode:
#     def __init__(self, val):
#         self.val = val
#         self.left, self.right = None, None
#
#
# class Solution:
#
#     def array_2_bst(self, data):
#         if len(data) == 1:
#             return TreeNode(data[0])
#
#         mid = len(data) // 2
#         root = TreeNode(data[mid])
#
#         root.left = self.array_2_bst(data[:mid])
#
#         root.right = self.array_2_bst(data[mid+1:])
#
#         return self.root
#
#
#     def preorder_traverse(self, node):
#         if not node:
#             return []
#
#         res = []
#         res.append(node.val)
#         res += self.preorder_traverse(node.left)
#         res += self.preorder_traverse(node.right)
#
#         return res
#
#
#     def solution(self, array):
#         if not array:
#             return []
#
#         root = self.array_2_bst(array)
#         return self.preorder_traverse(root)
#
#

# Logs table:
# +------------+
# | log_id     |
# +------------+
# | 1          |
# | 2          |
# | 3          |
# | 7          |
# | 8          |
# | 10         |
# +------------+
#
# Result table:
# +------------+--------------+
# | start_id   | end_id       |
# +------------+--------------+
# | 1          | 3            |
# | 7          | 8            |
# | 10         | 10           |
# +------------+--------------+
# The result table should contain all ranges in table Logs.
# From 1 to 3 is contained in the table.
# From 4 to 6 is missing in the table
# From 7 to 8 is contained in the table.
# Number 9 is missing in the table.
# Number 10 is contained in the table.
#
#
# 1  1  0
# 2  2  0
# 3  3  0
# 7  4  3
# 8  5  3
# 10 6  4
#
# select min(log_id) over (partition by flag) as start_id, max(log_id) over (partition by flag) as end_id
# from
# (select log_id, row_number() over (order by log_id) as rk, log_id - rk as flag
# from Logs)
# group by flag;
#
#
# select distinct min(log_id) over (partition by flag) as start_id, max(log_id) over (partition by flag) as end_id
# from
# (select log_id, row_number() over (order by log_id) as rk, log_id - rk as flag
# from development.logs)
# group by flag, log_id;
#
# ---------------------
#
#
# Taxi Company would like to design a data model to capture all critical data elements and capture some of the critical business questions.
#
# Example Quuestions:
#
# Track rides done by driver and their Performance
# How many rides are happening to a common/famous destinations each day( Airports , Parks , Museums etc)
# How many trips are cancelled per day.
# How many rides and the average price during the peak hour per day.
#
#
# create table rides (
#     ride_id  int,
#              start_time  datatime,
#                          end_time    datetime,
#                                      destination  varchar(100),
#                                                   start        varchar(100)
# );
#
# create table trips (
#     trip_id  int,
#              start_time  datatime,
#                          end_time    datetime,
#                                      status   boolean
# );
#
# create table prices (
#     ride_id   int,
#               start_time  datatime,
#                           end_time    datetime,
#                                       price     float
# );
#
#
# ----------------------------------
#
# Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.
#
# Example 1:
#
# Input: nums = [0,1,0,3,12]
# Output: [1,3,12,0,0]
#
#
# class Solution:
#     def move_0_to_end(self, nums):
#         res = []
#         cnt = 0
#         for i in range(len(nums)):
#             if nums[i] == 0:
#                 cnt += 1
#             else:
#                 res.append(nums[i])
#         res += [0] * cnt
#         return res


"""
1. ask clarifying questions - gather requirements (scale, performance, APIs), documenting requirements right away
   scale: how many DAU users? transactions per second? Queries per second (QPS) API supports 500 transactions/second?
   functionalities: what functionalities do I need to implement?
   
2. translate requirements into application / system design
3. start with requirement analysis
4. propose meaningful functionalities / services
5. UI front end (button invokes an API) - 
   web server, APIs (how the data is gonna be served) representation / services (payment, order) - 
   database schema data model (entities, key attributes, relationships, security)
6. API: Domain/resource/{parameters} - 
   Rest API uses HTTP requests to GET, POST, PUT (update) and DELETE data.
   HTTP: a request-response protocol between a client and server. 
   http request POST to insert，send data (JSON object) to server (RDS) to create/update
   http request GET to request data from the server (RDS)
   Example: A client (browser) sends an HTTP request to the server; then the server returns a response to the client.
7. architecture - tradeoff
8. audience / users - security (internal / external) - reliability (consistency of response) - 
   recovery (crash in-between and restart from last) - monitoring / logging (at different APIs)
9. ask for feedback - discuss options, pros and cons
10. focus on customer satisfaction and user experience
"""

"""
 
"""

# Q: Process user activity events files, set up ingestion, build dataset that track logins (a specific type of event).
#
# Source: daily data dump from the service, contains events from the beginning of the previous day till a certain time in current day.
# ```
# user_activity_yyyy-mm-dd HH:MM:SS.csv
# (event_time; user_id; event_type, event_details, result)
# ```
#
# day  user how many events login?
# time - day
# group by 1, 2, count(distinct successful logins)
#
# df.coalesce(k).write("s3://pathtofoloder/ds=20211103/") 200GB
#
# 200MB per partition
#
# 200GB / 200MB = k
#
# parquet
# json
#
# write
# read
#
#
#
#
# For the following input:
# ```
# Server StartDate EndDate
# A 01/12/2020 5/20/2020
# B 04/01/2020 10/10/2020
# ...
# Z 05/01/2020 12/11/2020
# A1 99/02/2020 12/01/2020
# A 11/12/2020 12/20/2020
# ```
#
# write function to validate data and generate the monthly activity report, put records that failed validation into error list
#
# error_list (invalid file records)
# ```
# 'A1 99/02/2020 12/01/2020'
# ```
#
# ```
# monthly_report
# Year_Month Active_Servers New_Active_Servers
# 2020_01 1 1
# 2020_02 1 0
# ...
# 2020_04 2 1
# 2020_05 3 1
# ```
#
#
# def solution(input):
#     errorList =  []
#     tot = []
#     for row in input:
#         server = row['Server']
#         start = row['StartDate']
#         end = row['EndDate']
#         start_month, end_month = start.split('/')[1], end.split('/')[1]
#         start_year, end_year = start.split('/')[-1], end.split('/')[-1]
#         if start_month < 1 or start_month > 12:
#             errorList.append(row)
#         if end_month < 1 or end_month > 12:
#             errorList.append(row)
#         if start_month in ('1', '3', '5', '7', '8' ...) and start_day > 31:
#             errorList.append(row)
#
#         for month in range(int(start_month), int(end_month) + 1):
#             tot.append(start_year + '_' + str(month))
#     res = []
#     for m in set(tot):
#         Active_Servers = tot.count(m)
#         Year_Month = m
#         New_Active_Servers = tot.countFirst()
#         res.append((Year_Month, Active_Servers, New_Active_Servers))
#
#     return res # array of tuples



#
# Lets say we have an HR system with lots of downstream data consumers. Everytime somebody changes their name, it needs to be reflected in these other systems. How would you design such integration?
#
# HR -> data - middleware -> upstream data source to downstream systems
#
#
#
#
#
# Data model
#
# Entities attributes
#
#
#
# employee
#
# employeeID, name, salary, modified_at
#
#
#
#
# Name
# John - johnny
#
# Ds System 1 resquest: ts, name, dob, address
# Job will export df.select(ts, name, dob, address).filter(modified_at = ‘2021-11-14’).write(path)
# Path s3://path/2021/11/14/data.json
#
# Run daily
#
# domain/resource/parameter
#
# ourHRsystem.com/v1/employee/1/modifiedAt=xxxx&modifiedAt=xxxx




















