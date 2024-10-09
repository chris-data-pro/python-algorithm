# Write a function


"""
Given n steps to reach to the top
Each time you can either climb 1 or 2 steps.
Return how many distinct ways can you climb to the top
"""

"""
@param n: integer
@return: integer number of ways
"""
def number_of_ways_bf(n):  # brute force
    if not n:
        return 0
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2

    return number_of_ways_bf(n - 1) + number_of_ways_bf(n - 2)


"""
@param n: integer
@return: integer number of ways
"""
def number_of_ways_dp(n):
    if not n:
        return 0
    if n <= 0:
        return 0

    res = [0, 1, 2]
    for i in range(3, n + 1):
        res.append(res[i - 1] + res[i - 2])

    return res[n]


"""
Fibonacci 
"""
def fibonacci_dp(n):  # Time Complexity: O(n), Space Complexity: O(1)
    if n == 0:
        return 0
    if n == 1:
        return 1
    last_last = 0
    last = 1
    for i in range(2, n + 1):
        current = last_last + last
        last_last = last
        last = current
    return current


if __name__ == '__main__':
    # cs = ClimbStairs()
    print(number_of_ways_dp(39))
    for n in range(10):
        print(fibonacci_dp(n))
