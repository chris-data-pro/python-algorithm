# Write a function to calculate change for a given number


class NumberOfChanges:
  """
  Given a number of cents
  Return list of numbers [number of quarters, number of dimes, number of nickels, number of pennies]
  """

  """
  @param amount: Given an integer
  @return: list of integers
  """
  def number_of_changes(self, amount):
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
    if residual == 0:
      return 0
    return residual // coin, residual % coin
