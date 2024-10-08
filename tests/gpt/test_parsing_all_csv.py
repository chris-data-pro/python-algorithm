import os
import shutil
import unittest
from pathlib import Path
from gpt.parsing_all_csv import parsing_all_csv


class TestParsingAllCsv(unittest.TestCase):
    def setUp(self):
        self.input_course_dir = 'test_Work/test_course'

    def test_parsing_all_json_only_m1(self):
        # create data path directory if not exists
        Path(self.input_course_dir).mkdir(parents=True, exist_ok=True)
        Path(self.input_course_dir + '/m1').mkdir(parents=True, exist_ok=True)

        parsing_all_csv(self.input_course_dir)
        self.assertTrue('test_course' in os.listdir('data/'))
        self.assertTrue('m1_content.txt' in os.listdir('data/test_course/'))
        self.assertTrue(os.stat('data/test_course/m1_content.txt').st_size == 0)

        # remove dummy folders
        shutil.rmtree('test_Work')
        shutil.rmtree('data')

    def test_parsing_all_json_m1_vtt(self):
        # create data path directory if not exists
        Path(self.input_course_dir).mkdir(parents=True, exist_ok=True)
        Path(self.input_course_dir + '/m1').mkdir(parents=True, exist_ok=True)
        Path(self.input_course_dir + '/English VTT Files').mkdir(parents=True, exist_ok=True)

        parsing_all_csv(self.input_course_dir)
        self.assertTrue('test_course' in os.listdir('data/'))
        self.assertTrue('m1_content.txt' in os.listdir('data/test_course/'))
        self.assertTrue(len(os.listdir('data/test_course/')) == 1)

        # remove dummy folders
        shutil.rmtree('test_Work')
        shutil.rmtree('data')


if __name__ == '__main__':
    unittest.main()
