import unittest
from algorithm.best_buy_sell_stock import BestBuySellStock


class TestBestBuySellStock(unittest.TestCase):
    def setUp(self):
        self.bbss = BestBuySellStock()

        self.input_1 = [4, 3, 7, 1, 5]
        self.result_1_once = (3, 4)
        self.result_1_multiple = [(1, 2), (3, 4)]

        self.input_2 = [4, 3, 8, 1, 5]
        self.result_2_once = (1, 2)
        self.result_2_multiple = [(1, 2), (3, 4)]

        self.input_3 = [4, 1, 5, 3, 7]
        self.result_3_once = (1, 4)
        self.result_3_multiple = [(1, 2), (3, 4)]

        self.input_4 = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        self.result_4_once = ()
        self.result_4_multiple = []

        self.input_5 = [12, 6]
        self.result_5_once = ()
        self.result_5_multiple = []

        self.input_6 = [5, 3, 8, 8, 55, 38, 41, 1, 42, 54, 50, 60, 20, 12, 1, 28]
        self.result_6_once = (7, 11)
        self.result_6_multiple = [(1, 4), (5, 6), (7, 9), (10, 11), (14, 15)]

        self.input_7 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        self.result_7_once = ()
        self.result_7_multiple = []

        self.input_8 = [3, 8, 8, 55, 38, 1, 7, 42, 54, 53]
        self.result_8_once = (5, 8)
        self.result_8_multiple = [(0, 3), (5, 8)]

    def test_buy_sell_once(self):
        self.assertEqual(self.bbss.buy_sell_once(self.input_1), self.result_1_once)
        self.assertEqual(self.bbss.buy_sell_once(self.input_2), self.result_2_once)
        self.assertEqual(self.bbss.buy_sell_once(self.input_3), self.result_3_once)
        self.assertEqual(self.bbss.buy_sell_once(self.input_4), self.result_4_once)
        self.assertEqual(self.bbss.buy_sell_once(self.input_5), self.result_5_once)
        self.assertEqual(self.bbss.buy_sell_once(self.input_6), self.result_6_once)
        self.assertEqual(self.bbss.buy_sell_once(self.input_7), self.result_7_once)
        self.assertEqual(self.bbss.buy_sell_once(self.input_8), self.result_8_once)

    def test_buy_sell_multiple(self):
        self.assertEqual(self.bbss.buy_sell_multiple(self.input_1), self.result_1_multiple)
        self.assertEqual(self.bbss.buy_sell_multiple(self.input_2), self.result_2_multiple)
        self.assertEqual(self.bbss.buy_sell_multiple(self.input_3), self.result_3_multiple)
        self.assertEqual(self.bbss.buy_sell_multiple(self.input_4), self.result_4_multiple)
        self.assertEqual(self.bbss.buy_sell_multiple(self.input_5), self.result_5_multiple)
        self.assertEqual(self.bbss.buy_sell_multiple(self.input_6), self.result_6_multiple)
        self.assertEqual(self.bbss.buy_sell_multiple(self.input_7), self.result_7_multiple)
        self.assertEqual(self.bbss.buy_sell_multiple(self.input_8), self.result_8_multiple)


if __name__ == '__main__':
    unittest.main()
