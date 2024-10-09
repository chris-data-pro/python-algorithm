import unittest
from algorithm.dp_numbers import DPClimbStairs


class TestDPClimbStairs(unittest.TestCase):
    def setUp(self):
        self.cs = DPClimbStairs()

        self.input_1 = 39
        self.result_1 = 102334155

        self.input_2 = 3
        self.result_2 = 3

        self.input_3 = 0
        self.result_3 = 0

        self.input_4 = -10
        self.result_4 = 0

    def test_number_of_ways(self):
        self.assertEqual(self.cs.number_of_ways(self.input_1), self.result_1)
        self.assertEqual(self.cs.number_of_ways(self.input_2), self.result_2)
        self.assertEqual(self.cs.number_of_ways(self.input_3), self.result_3)
        self.assertEqual(self.cs.number_of_ways(self.input_4), self.result_4)


if __name__ == '__main__':
    unittest.main()
