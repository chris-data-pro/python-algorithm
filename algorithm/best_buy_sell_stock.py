import heapq


class BestBuySellStock:
    """
    Given an array for which the ith element is the price of a stock on day i.
    For different rules, return best Buying and Selling day for maximum gain
    """

    """
    149
    Rules: 1. Buy first 2. only buy once and sell once
    @param prices: Given an integer array
    @return: tuple - the best Buying and Selling day for maximum gain
    """
    def buy_sell_once(self, prices):
        if not prices or len(prices) < 2 or len(set(prices)) <= 1:
            return ()

        buy, profit = 0, 0
        transactions = {}

        for i in range(1, len(prices)):
            if prices[buy] > prices[i]:
                buy = i
            else:
                new = prices[i] - prices[buy]
                if new >= profit:
                    transactions[(buy, i)] = new
                profit = max(profit, new)

        return list(transactions.keys())[-1] if transactions else ()

    def buy_sell_once_dfs(self, prices):
        res, memo = 0, {}
        for i in range(len(prices)):
            res = max(res, self.dfs(prices, i, 0, 0, memo, 1))
        return res

    """
    150
    Rules: 1. buy first and then sell 2. Buy and sell as many times as possible.
    @param prices: Given an integer array
    @return: list of tuples - the best Buying and Selling day for maximum gain
    """
    def buy_sell_multiple(self, prices):
        if not prices or len(prices) < 2 or len(set(prices)) <= 1:
            return []

        buy, profit = 0, 0
        transactions = {}

        for i in range(1, len(prices)):
            if prices[i - 1] > prices[i]:
                if i - 1 > buy: transactions[(buy, i - 1)] = profit
                buy = i

            profit = prices[i] - prices[buy]

        if i > buy:
            transactions[(buy, i)] = profit
        return list(transactions.keys())

    def buy_sell_multiple_dfs(self, prices):
        res, memo = 0, {}
        for i in range(len(prices)):
            res = max(res, self.dfs(prices, i, 0, 0, memo, float('inf')))
        return res

    """
    151
    Rules: 1. buy first and then sell 2. Buy and sell at most twice
    """
    def buy_sell_twice_dfs(self, prices):
        res, memo = 0, {}
        for i in range(len(prices)):
            res = max(res, self.dfs(prices, i, 0, 0, memo, 2))
        return res

    def dfs(self, prices, index, state, count, memo, k):
        if (index, state, count) in memo:
            return memo[(index, state, count)]

        if index >= len(prices) or count >= k:  # at most k transactions
            return 0

        case1, case2, case3, case4 = 0, 0, 0, 0

        if state == 0:  # 当前手中没有股票
            case1 = self.dfs(prices, index + 1, 0, count, memo, k)  # 不买当前的，状态不变, count不变
            case2 = -prices[index] + self.dfs(prices, index + 1, 1, count, memo, k)  # 买当前的，状态变1, count不变

        else:  # state == 1 当前手中持有股票
            case3 = prices[index] + self.dfs(prices, index + 1, 0, count + 1, memo, k)  # 卖当前的，状态变0, count加1
            case4 = self.dfs(prices, index + 1, 1, count, memo, k)  # 不卖当前的，状态不变, count不变

        memo[(index, state, count)] = max(case1, case2, case3, case4)  # (index, state, count) -> max profit

        return memo[(index, state, count)]

    """
    1691
    Rules: 1. can only trade at most once a day 2. don't have to sell before buy new stock 3. trade any times
    @return: the maximum profit
    """
    def max_profit(self, prices):
        minheap = []
        res = 0
        for price in prices:
            if minheap and price > minheap[0]:  # 如果price比之前遇到过的最低价高
                res += price - heapq.heappop(minheap)  # 收益就是当前price - 遇到过的最低价
                heapq.heappush(minheap, price)
            heapq.heappush(minheap, price)  # 同时将当前值压入队列
        return res


if __name__ == '__main__':
    bbss = BestBuySellStock()
    print(bbss.buy_sell_once([4, 3, 7, 1, 5]))  # expect (3, 4)
    print(bbss.buy_sell_once_dfs([4, 3, 7, 1, 5]))
    print(bbss.buy_sell_multiple([4, 3, 7, 1, 5]))
    print(bbss.buy_sell_multiple_dfs([4, 3, 7, 1, 5]))
    print(bbss.buy_sell_multiple([1, 2, 4, 2, 5, 7, 2, 4, 9, 0]))
    print(bbss.buy_sell_multiple_dfs([1, 2, 4, 2, 5, 7, 2, 4, 9, 0]))
    print(bbss.buy_sell_twice_dfs([1, 2, 4, 2, 5, 7, 2, 4, 9, 0]))  # expect 13
    print(bbss.max_profit([29, 52, 12, 51, 8, 38, 77, 10, 54, 90, 26, 8, 13, 97, 40, 96, 87, 80]))  # expect 500
