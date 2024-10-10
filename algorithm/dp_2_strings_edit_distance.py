# Write a function
import heapq
import math


"""
Given 2 strings
3 types of edits are permitted on a word: 1. Insert a character 2. Delete a character 3. Replace a character
Return at least how many edits are required to convert str1 to str2
"""


"""
@param str1: string
@param str2: string
@return: integer the edit distance
"""
def least_edit_distance_bf(word1, word2):  # brute force
    def dp(i, j):
        if i == 0:
            return j
        if j == 0:
            return i
        if word1[i - 1] == word2[j - 1]:
            return dp(i - 1, j - 1)
        else:
            return min(dp(i, j - 1) + 1,
                       dp(i - 1, j) + 1,
                       dp(i - 1, j - 1) + 1)

    return dp(len(word1), len(word2))


def least_edit_distance(word1, word2):  # O(m * n)
    m, n = len(word1), len(word2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]  # (n + 1) rows X (m + 1) cols - list of lists

    for x in range(n + 1):
        dp[x][0] = x

    for y in range(m + 1):
        dp[0][y] = y

    for i in range(1, n + 1):  # ith row
        for j in range(1, m + 1):  # jth col
            if word1[j - 1] == word2[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i][j - 1] + 1,
                               dp[i - 1][j] + 1,
                               dp[i - 1][j - 1] + 1)
    # print(dp)
    return dp[-1][-1]


"""
640
@param s: a string
@param t: a string
@return: true if they are one edit distance apart or false
"""
def is_one_edit_distance(s, t):  # O(max(ls, lt))
    ls, lt = len(s), len(t)
    if ls == lt:
        diff = 0
        for i in range(ls):
            if s[i] != t[i]:
                diff += 1
        return diff == 1

    if ls == lt + 1:
        for i in range(lt):
            if s[i] != t[i]:
                return s[i + 1:] == t[i:]
        return True

    if lt == ls + 1:
        for i in range(ls):
            if s[i] != t[i]:
                return s[i:] == t[i + 1:]
        return True

    return False


"""
Given a list of words 'L' and a target word 'w'
Return the top 1 word in 'L' that are closest to the target word.
@param str1: list of string
@param str2: string
@return: the smallest ed, and that word
"""
def sort_smallest(L, w):  # O(NX), X is the complexity of the mapping
    if not L:
        return

    word = L[0]
    distance = least_edit_distance(word, w)

    for i in range(1, len(L)):
        d = least_edit_distance(L[i], w)
        if d < distance:
            distance = d
            word = L[i]

    return distance, word


"""
Given a list of words 'L' and a target word 'w'
Return the top 1 word in 'L' that are farthest to the target word.
@param str1: list of string
@param str2: string
@return: the word with largest ed, and its ed
"""
def sort_largest(L, w):  # O(NX), X is the complexity of the mapping
    if not L:
        return

    word = L[0]
    distance = least_edit_distance(word, w)

    for i in range(1, len(L)):
        d = least_edit_distance(L[i], w)
        if d > distance:
            distance = d
            word = L[i]

    return distance, word


def sort_largest_k(L, w, k):  # O(NX + NlogN)
    if not L:
        return

    heap = []
    for l in L:
        d = least_edit_distance(l, w)
        if len(heap) < k:
            heapq.heappush(heap, (d, l))
        else:
            heapq.heappushpop(heap, (d, l))  # Push item, then pop and return the smallest item from the heap

    return heap


def sort_smallest_k(L, w, k):  # O(NX + NlogN + k)
    if not L:
        return

    heap = []
    for l in L:
        d = least_edit_distance(l, w)
        if len(heap) < k:
            heapq.heappush(heap, (-d, l))
        else:
            heapq.heappushpop(heap, (-d, l))

    return [(-x[0], x[1]) for x in heap]


"""
Given a list of words 'L' and a target word 'w'
Return the top 5 words in 'L' that are closest to the target word.
@param str1: list of string
@param str2: string
@return: sub list of 'L'
"""
def sort_tops(L, w):  # O(NX + NlogN + k)
    res = {}
    for l in L:
        res[least_edit_distance(l, w)] = l  # O(NX)

    top5 = sorted(res.items(), key=lambda item: item[0])[:5]  # Time Complexity O(NlogN + 5)

    return top5


def heap_tops(L, w):  # O(NX + NlogN + klogN)
    res = []
    for l in L:
        heapq.heappush(res, (least_edit_distance(l, w), l))  # O(N * (X + logN))
        # if len(res) > 5:
        #     heapq.heappop(res)

    # return heapq.nsmallest(5, res)
    return [heapq.heappop(res) for _ in range(5)]  # O(5logN)


if __name__ == '__main__':
    # ed = DPEditDistance()
    print(least_edit_distance("horse", "ros"))  # expect 3
    print(least_edit_distance_bf("intention", "execution"))  # expect 5
    print(least_edit_distance("aDb", "adb"))  # expect 1
    print(least_edit_distance("trinitrophenylmethylnitramine", "dinitrophenylhydrazine"))  # expect 10
    print(sort_tops(["aaaa", "aa", "a", "aaa", "aaaaa", "aaaaaaa"], "a"))
    print(heap_tops(["aaaa", "aa", "a", "aaa", "aaaaa", "aaaaaaa"], "a"))
    print(sort_smallest(["aaaa", "aa", "a", "aaa", "aaaaa", "aaaaaaa"], "a"))  # expected (0, 'a')
    print(sort_largest(["aaaa", "aa", "a", "aaa", "aaaaa", "aaaaaaa"], "a"))  # expected (6, 'aaaaaaa')
    print(sort_largest_k(["aaaa", "aa", "a", "aaa", "aaaaa", "aaaaaaa"], "a", 5))
    print(sort_smallest_k(["aaaa", "aa", "a", "aaa", "aaaaa", "aaaaaaa"], "a", 5))


"""
Design Google Search 'Did you mean' feature:
  You are given a list of words 'L' and a word that user has input 'w'. 
  You have to give the top 5 suggestions that are closest to the input word.

  Closeness of two words is defined with edit distance. eg, 'bread' and 'redding'

  bread -> read (remove 'b')
  read -> redd (replace 'a' with 'd')
  redd -> reddi (add 'i')
  reddi -> reddin (add 'n')
  reddin -> redding (add 'g')

  That is atleast 5 edits

  Sample:
  L = ["redding", "reader", "breeder", "baker", "meadow", "meat"]
  w = "bread"
  distances = [5, 3, 3, 4, 4, 3]
  result = [reader, breeder, meat, baker, meadow]


  F(bread, redding) = ?
  F(brea, reddin) = replace
  F(brea, redding) = add
  F(bread, reddin) = removing

"""

"""
import unittest


class Solution:
    def myFunc(self, L, w):
        # code here
        res = {}
        for l in L:
            res[l] = self.computeDistance(l, w)

        top5 = sorted(res.items(), key=lambda item: item[1])[:5]

        return [x[0] for x in top5]

    def computeDistance(self, W1, W2):
        if not W1 and not W2:
            return 0
        if not W1:
            return len(W2)
        if not W2:
            return len(W1)
        if W1 == W2:
            return 0

        distance = {}

        return self.dfs(W1, W2, distance)

    def dfs(self, W1, W2, distance):
        l1, l2 = len(W1), len(W2)

        if not W1 and not W2:
            return 0
        if not W1:
            return l2
        if not W2:
            return l1
        if W1 == W2:
            return 0
        if (l1, l2) in distance:
            return distance[(l1, l2)]

        answer = 0

        if W1[l1 - 1] == W2[l2 - 1]:
            answer = self.dfs(W1[:l1 - 1], W2[:l2 - 1], distance)
        else:
            answer = min(self.dfs(W1[:l1 - 1], W2[:l2 - 1], distance) + 1,
                         self.dfs(W1[:l1 - 1], W2, distance) + 1,
                         self.dfs(W1, W2[:l2 - 1], distance) + 1)

        distance[(l1, l2)] = answer

        return answer

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.s = Solution()
        self.input_11 = "abc"
        self.input_12 = "ab"
        self.result_1 = 1
        self.input_21 = "bread"
        self.input_22 = "redding"
        self.result_2 = 5
        self.input_31 = "thanka"
        self.input_32 = "thanks"
        self.result_3 = 1

    def testMyFunc(self):
        self.assertEqual(self.s.computeDistance(self.input_11, self.input_12), self.result_1)
        self.assertEqual(self.s.computeDistance(self.input_21, self.input_22), self.result_2)
        self.assertEqual(self.s.computeDistance(self.input_31, self.input_32), self.result_3)


if __name__ == '__main__':
    unittest.main()
"""
