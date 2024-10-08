import os
import re
import csv
import json
import docx2txt
from pathlib import Path
from helper import video_transcript_2_txt
from common import var_config


def parsing_all_docx(course_dir, data_dir):
    """Parses Course metadata and Q&A Text Contents From All Modules in word documents .docx format.

    Parameters
    ----------
    course_dir : str, required
        The path of the course.
    data_dir : str, required
        The path of the output txt data.
    """

    course = course_dir.split('/')[-1]

    print('Parsing {} Course metadata and Q&A Text Contents From All Modules'.format(course))

    # create data path directory if not exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for doc in os.listdir(course_dir + '/word-docs'):

        meta_file = data_dir + '/' + doc.replace('.docx', '.txt')

        with open(meta_file, "w") as output_file:
            output_file.write(docx2txt.process(course_dir + '/word-docs/' + doc).replace(' ', ''))
            output_file.close()

        # text = docx2txt.process(course_dir + '/word-docs/' + doc)
        # print(text)


if __name__ == '__main__':
    course = var_config.course_dir.split('/')[-1]
    parsing_all_docx(var_config.word_course_dir, var_config.cwd + '/data/' + course)
