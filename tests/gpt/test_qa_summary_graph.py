import os
import shutil
import unittest
from pathlib import Path
from gpt.qa_summary_graph import get_query_configs


class TestQASummaryGraph(unittest.TestCase):
    def setUp(self):
        self.key1 = 'index_struct_type'
        self.key2 = 'query_mode'
        self.key3 = 'query_kwargs'

    def test_get_query_configs(self):

        qcs = get_query_configs()
        for qc in qcs:
            self.assertTrue(self.key1 in qc)
            self.assertTrue(self.key2 in qc)
            self.assertTrue(self.key3 in qc)


if __name__ == '__main__':
    unittest.main()
