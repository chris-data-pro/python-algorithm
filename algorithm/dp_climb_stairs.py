# Write a function


class DPClimbStairs:
    """
    Given n steps to reach to the top
    Each time you can either climb 1 or 2 steps.
    Return how many distinct ways can you climb to the top
    """

    """
    @param n: integer
    @return: integer number of ways
    """
    def number_of_ways_bf(self, n):
        if not n:
            return 0
        if n <= 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 2

        return self.number_of_ways_bf(n - 1) + self.number_of_ways_bf(n - 2)

    """
    @param n: integer
    @return: integer number of ways
    """
    def number_of_ways(self, n):
        if not n:
            return 0
        if n <= 0:
            return 0

        res = [0, 1, 2]
        for i in range(3, n + 1):
            res.append(res[i - 1] + res[i - 2])

        return res[n]


# if __name__ == '__main__':
#     cs = ClimbStairs()
#     print(cs.number_of_ways(39))
