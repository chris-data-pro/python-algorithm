# Write a function to find out the best Buying and Selling day for maximum gain from daily stock prices


class BestBuySellStock:
  """
  Given an array for which the ith element is the price of a stock on day i.
  For different rules, return best Buying and Selling day for maximum gain
  """

  """
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
        if new >= profit: transactions[(buy, i)] = new
        profit = max(profit, new)

    return list(transactions.keys())[-1] if transactions else ()


  """
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

    if i > buy: transactions[(buy, i)] = profit
    return list(transactions.keys())
