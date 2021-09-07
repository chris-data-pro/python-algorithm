import unittest
from algorithm.number_of_changes import NumberOfChanges


class TestNumberOfChanges(unittest.TestCase):
  def setUp(self):
    self.noc = NumberOfChanges()

    self.input_1 = 83
    self.result_1 = [3, 0, 1, 3]

    self.input_2 = 139
    self.result_2 = [5, 1, 0, 4]

    self.input_3 = 21
    self.result_3 = [0, 2, 0, 1]

    self.input_4 = 0
    self.result_4 = []

    self.input_5 = 1
    self.result_5 = [0, 0, 0, 1]

    self.input_6 = -5
    self.result_6 = []

    self.input_7 = 100
    self.result_7 = [4, 0, 0, 0]

    self.input_8 = 80
    self.result_8 = [3, 0, 1, 0]

  def test_number_of_changes(self):
    self.assertEqual(self.noc.number_of_changes(self.input_1), self.result_1)
    self.assertEqual(self.noc.number_of_changes(self.input_2), self.result_2)
    self.assertEqual(self.noc.number_of_changes(self.input_3), self.result_3)
    self.assertEqual(self.noc.number_of_changes(self.input_4), self.result_4)
    self.assertEqual(self.noc.number_of_changes(self.input_5), self.result_5)
    self.assertEqual(self.noc.number_of_changes(self.input_6), self.result_6)
    self.assertEqual(self.noc.number_of_changes(self.input_7), self.result_7)
    self.assertEqual(self.noc.number_of_changes(self.input_8), self.result_8)


if __name__ == '__main__':
  unittest.main()
