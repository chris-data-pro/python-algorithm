import unittest
from algorithm.dp_2_strings import DPLongestCommonSubsequence


class TestDPLongestCommonSubsequence(unittest.TestCase):
    def setUp(self):
        self.lcs = DPLongestCommonSubsequence()

        self.input_11 = "a"
        self.input_12 = "aaaaaaaaaaa"
        self.result_1 = 1

        self.input_21 = "daabeddbcedeabcbcbec"
        self.input_22 = "daceeaeeaabbabbacedd"
        self.result_2 = 10

        self.input_31 = ""
        self.input_32 = ""
        self.result_3 = 0

        self.input_41 = ""
        self.input_42 = "bs"
        self.result_4 = 0

    def test_length_of_lcs(self):
        self.assertEqual(self.lcs.length_of_lcs(self.input_11, self.input_12), self.result_1)
        self.assertEqual(self.lcs.length_of_lcs(self.input_21, self.input_22), self.result_2)
        self.assertEqual(self.lcs.length_of_lcs(self.input_31, self.input_32), self.result_3)
        self.assertEqual(self.lcs.length_of_lcs(self.input_41, self.input_42), self.result_4)


if __name__ == '__main__':
    unittest.main()
