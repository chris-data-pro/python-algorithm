import time
import math
import random
from itertools import permutations, combinations

"""
# Divide 10 balls randomly into 12 boxes.
# What's the probability that exactly 10 boxes are empty?
from Donny Chen (Cisco) to Everyone:    12:03  PM
# Question 1
# Write a simulation program to find out the probability.
# Helpers: import random    , random.randint(a, b) returns a random integer N such that a <= N <= b   -- you can search for numpy classes and methods
# Please provide the solution below
"""


def simu(n):

    res, ct = [], 0
    for _ in range(n):
        for _ in range(10):
            res.append(random.randint(0, 11))
        if len(set(res)) == 2:
            ct += 1
    return ct / n


if __name__ == '__main__':
    p = math.comb(12, 2) * (2 / 12) ** 10 * (10 / 12) ** 2
    print(p)


    # C(12, 2) * (2/12) ** 2 * (10/12) ** 10
    # 12 ** 10

    # if __name__ == '__main__':
    #     import random
    #     boxes = range(12)
    #
    #     # res = []
    #     # for _ in range(10):
    #     #     res.append(random.randint(0, 11))
    #
    #     print(random.randint(0, 11))
    #     # expected = True
    #
    #     # assert udf1(puzzle) == expected
    #     # print(puzzle)
    def rand(input_int):
        random = int(time.time()*1000)
        return random % input_int

    print(rand(1000))
