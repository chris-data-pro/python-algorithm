import unittest
from algorithm.dp_2_strings_edit_distance import DPEditDistance


class TestDPEditDistance(unittest.TestCase):
    def setUp(self):
        self.ed = DPEditDistance()

        self.input_11 = "a"
        self.input_12 = "aaaaaaaaaaa"
        self.result_1 = 10

        self.input_21 = "horse"
        self.input_22 = "ros"
        self.result_2 = 3

        self.input_31 = ""
        self.input_32 = ""
        self.result_3 = 0

        self.input_41 = ""
        self.input_42 = "bs"
        self.result_4 = 2

        self.input_51 = "intention"
        self.input_52 = "execution"
        self.result_5 = 5

        self.input_61 = "trinitrophenylmethylnitramine"
        self.input_62 = "dinitrophenylhydrazine"
        self.result_6 = 10

        self.input_71 = ""
        self.input_72 = ""
        self.result_7 = False

        self.input_81 = "trinitrophenylmethylnitramine"
        self.input_82 = "trinitrophenylmethylntitramine"
        self.result_8 = True

    def test_length_of_lcs(self):
        self.assertEqual(self.ed.least_edit_distance(self.input_11, self.input_12), self.result_1)
        self.assertEqual(self.ed.least_edit_distance(self.input_21, self.input_22), self.result_2)
        self.assertEqual(self.ed.least_edit_distance(self.input_31, self.input_32), self.result_3)
        self.assertEqual(self.ed.least_edit_distance(self.input_41, self.input_42), self.result_4)
        self.assertEqual(self.ed.least_edit_distance(self.input_51, self.input_52), self.result_5)
        self.assertEqual(self.ed.least_edit_distance(self.input_61, self.input_62), self.result_6)

    def test_sort_tops(self):
        self.assertEqual(self.ed.sort_tops(["aaaa", "aa", "a", "aaa", "aaaaa", "aaaaaaa"], "a"),
                         ["a", "aa", "aaa", "aaaa", "aaaaa"])

    def test_heap_tops(self):
        self.assertEqual(self.ed.heap_tops(["aaaa", "aa", "a", "aaa", "aaaaa", "aaaaaaa"], "a"),
                         ["a", "aa", "aaa", "aaaa", "aaaaa"])

    def test_is_one_edit_distance(self):
        self.assertEqual(self.ed.is_one_edit_distance(self.input_71, self.input_72), self.result_7)
        self.assertEqual(self.ed.is_one_edit_distance(self.input_81, self.input_82), self.result_8)


if __name__ == '__main__':
    unittest.main()
