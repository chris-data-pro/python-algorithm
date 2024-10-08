import os
import shutil
import unittest
from pathlib import Path
from gpt.helper import langchain_chunk
from gpt.qa_summary_graph import get_query_configs


class TestLangchainChunk(unittest.TestCase):
    def setUp(self):
        self.data_dir = '../../../gpt/data/pe1'
        self.reports_len = len(os.listdir(self.data_dir))

    def test_get_chunked_reports(self):
        crs = langchain_chunk.get_chunked_reports('../../../gpt/data/pe1')
        self.assertEqual(len(crs), self.reports_len)


if __name__ == '__main__':
    unittest.main()

