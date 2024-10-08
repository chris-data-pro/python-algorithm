import os
import shutil
import unittest
from pathlib import Path
from gpt.parsing_all_json import parsing_all_json


class TestParsingAllJson(unittest.TestCase):
    def setUp(self):
        self.input_course_dir = 'test_builds/test_course'

    def test_parsing_all_json_only_m1(self):
        # create data path directory if not exists
        Path(self.input_course_dir).mkdir(parents=True, exist_ok=True)
        Path(self.input_course_dir + '/m1').mkdir(parents=True, exist_ok=True)

        parsing_all_json(self.input_course_dir)
        self.assertTrue('test_course' in os.listdir('data/'))
        self.assertTrue('m1.txt' in os.listdir('data/test_course/'))
        self.assertTrue(os.stat('data/test_course/m1.txt').st_size == 0)

        # remove dummy folders
        shutil.rmtree('test_builds')
        shutil.rmtree('data')

    def test_parsing_all_json_m1_vtt(self):
        # create data path directory if not exists
        Path(self.input_course_dir).mkdir(parents=True, exist_ok=True)
        Path(self.input_course_dir + '/m1').mkdir(parents=True, exist_ok=True)
        Path(self.input_course_dir + '/English VTT Files').mkdir(parents=True, exist_ok=True)

        parsing_all_json(self.input_course_dir)
        self.assertTrue('test_course' in os.listdir('data/'))
        self.assertTrue('m1.txt' in os.listdir('data/test_course/'))
        self.assertTrue(len(os.listdir('data/test_course/')) == 1)

        # remove dummy folders
        shutil.rmtree('test_builds')
        shutil.rmtree('data')


if __name__ == '__main__':
    unittest.main()
