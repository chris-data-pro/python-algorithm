import random


class Probability:
    """
    Probability problems
    """

    """
    Given a list of city names and their corresponding populations, 
    write a function to output a random city name subject to the following constraint: 
    the probability to output a city's name is based on its population divided by the sum of all cities' population.
    For example:
    
    NY: 7
    SF: 5
    LA: 8
    The probability to generate NY is 7/20, SF is 1/4.
    
    """

    # input = {'NY': 7, 'SF': 5, 'LA': 8}
    # cities = []
    # populations = []

    def random_city_name(self, inputs):
        total = sum(inputs.values())
        r = random.random()
        print(r, r * total)

        lv = list(inputs.values())  # [5, 5, 8]
        sums = 0
        moving_sum = []
        for v in lv:
            sums += v
            moving_sum.append(sums)  # [5, 10, 18]
        print(moving_sum)

        for i in range(len(moving_sum)):  # 0 to len(moving_sum) - 1
            if r * total < moving_sum[i]:
                # return [item[0] for item in inputs.items() if item[1] == lv[i]][0]
                return list(inputs.keys())[i]

            # ith key-value from a dict

    # What if r * total = 6

    # input = {'NY': 5, 'SF': 5, 'LA': 8}
    # [5, 10, 18]
    # r * total = 9


if __name__ == '__main__':
    p = Probability()
    print(p.random_city_name({'NY': 5, 'SF': 5, 'LA': 8}))
