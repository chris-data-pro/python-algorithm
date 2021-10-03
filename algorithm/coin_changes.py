# Write a function to calculate change for a given number


class CoinChanges:
    """
    coin change problems
    """

    """
    Given a number of cents
    Return list of numbers [number of quarters, number of dimes, number of nickels, number of pennies]
    there are infinite number of each kind of coin 每个币种有无限多
    
    Input: 83
    Output: [3, 0, 1, 3] 三个25，零个10，一个5，三个1
    
    @param amount: Given an integer
    @return: list of integers
    """
    def coin_changes(self, amount):
        if not amount or amount < 0:
            return []

        residual = amount
        results = []
        coins = [25, 10, 5, 1]

        for coin in coins:
            n, residual = self.number_of_coin(residual, coin)
            results.append(n)

        return results

    """
    @param amount: Given an integer and a coin's value
    @return: number of coins, rest cents
    """
    def number_of_coin(self, residual, coin):
        return residual // coin, residual % coin

    """
    669
    given coins of different denominations and a total amount of money amount. 
    compute the fewest number of coins that you need to make up that amount. return -1 if cannot make up
    there are infinite number of each kind of coin 每个币种有无限多
    
    Input: [1, 2, 5], 11
    Output: 3 最少用两个5，一个1，共三个
    Input: [186,419,83,408], 6249
    Output: 20
    """
    def unlimited_coin_changes_fewest_number_coins(self, coins, amount):
        MAX = float('inf')
        ans = [MAX for i in range(amount + 1)]
        ans[0] = 0  # dp[i]为组成金额i需要的最小硬币数
        for i in range(1, amount + 1):
            for coin in coins:
                if i < coin:
                    continue
                ans[i] = min(ans[i], ans[i - coin] + 1)
        if ans[amount] == MAX:
            return -1
        return ans[amount]

    """
    each kind of coin has a max number to use
    
    Input: [1, 2, 5], [5, 2, 1], 11
    Output: 5 最少用了5个coins
    Explanation:
    11 = 5 + 2 * 2 + 2 * 1   => 5 coins used
    11 = 5 + 1 * 2 + 4 * 1   => 6 coins used
    """
    def limited_coin_change_fewest_number_coins(self, coins, nums, amount):
        inputs = []
        for c, num in reversed(list(zip(coins, nums))):  # 大面值在前 [5, 2, 2, 1, 1, 1, 1, 1]
            inputs += [c] * num
        if amount <= 0:
            return 0

        if sum(inputs) < amount:
            return -1
        elif sum(inputs) == amount:
            return len(inputs)

        if self.dfs_lcc(inputs, amount) == float('inf'):
            return -1
        else:
            return self.dfs_lcc(inputs, amount)

    def dfs_lcc(self, inputs, amount):  # returns inf if no way to make up amount
        if not inputs:
            return float('inf')
        for i, e in enumerate(inputs):
            if amount < e:
                min_cions = float('inf')
            elif amount == e:
                min_cions = 1
            else:
                min_cions = 1 + self.dfs_lcc(inputs[i + 1:], amount - e)

            if min_cions == float('inf'):
                continue
            else:
                return min_cions
        return float('inf')

    # def limited_coin_change(self, coins, nums, amount):
    #     dp = [0] + [float('inf')]*amount
    #     for c, num in zip(coins, nums):
    #         for j in range(1, num+1):
    #             for i in reversed(range(c*num, amount+1)):
    #                 dp[i] = min(dp[i], num + dp[i-c*j])
    #     print(dp)
    #     return dp[amount] if dp[amount] != float('inf') else -1

    """
    740
    given coins of different denominations and a total amount of money amount. 
    compute the number of combinations that make up that amount
    there are infinite number of each kind of coin 每个币种有无限多
    
    输入: amount = 8 和 coins = [2, 3, 8]
    输出: 3
    解释:
    有3种方法:
    8 = 8
    8 = 3 + 3 + 2
    8 = 2 + 2 + 2 + 2
    """
    def unlimited_coin_changes_number_ways(self, coins, amount):
        dp = [0] * (amount + 1)
        dp[0] = 1  # dp[i]表示能否取到两个集合之差为i的情况

        for coin in coins:
            for x in range(coin, amount+1):
                dp[x] += dp[x-coin]
        return dp[amount]


if __name__ == '__main__':
    cc = CoinChanges()
    print(cc.coin_changes(83))
    print(cc.coin_changes(100))
    print(cc.limited_coin_change_fewest_number_coins([1, 2, 5], [5, 2, 1], 6))
    print(cc.dfs_lcc([5, 2, 2, 1, 1, 1, 1, 1], 6))
