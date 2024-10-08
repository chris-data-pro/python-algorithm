import os
import re
import csv
from pathlib import Path
from gpt.helper import video_transcript_2_txt
from gpt.common import var_config


def parsing_all_csv(course_dir):
    """Parses Course Text Contents From All Modules in csv and Videos in vtt.

    Parameters
    ----------
    course_dir : str, required
        The path of the course.
    """

    course = course_dir.split('/')[-1]

    print('Parsing {} Course Text Contents From All Modules and Videos'.format(course))

    data_dir = 'data/' + course

    # create data path directory if not exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for module in os.listdir(course_dir):
        if module.startswith('m'):

            csv_file = course_dir + '/' + module + '/en-US/components.csv'

            txt_file = data_dir + '/' + module + '_content.txt'
            # print(txt_file)

            def striphtml(data):
                p = re.compile(r'<.*?>')
                return p.sub('', data.replace('&nbsp;', ' ').replace(' ', '').replace(' ', ''))

            with open(txt_file, "w") as my_output_file:
                # my_output_file.write(module.replace('m', 'Module') + '\n')
                if os.path.isfile(csv_file):
                    with open(csv_file, "r") as my_input_file:
                        [my_output_file.write("".join(striphtml(row[1])) + '\n')
                         for row in csv.reader(my_input_file)
                         if row[0].endswith('/title/') or row[0].endswith('/body/') or row[0]]
                my_output_file.close()

        if module == 'English VTT Files':
            for vtt in os.listdir(course_dir + '/English VTT Files'):

                transcript_file = data_dir + '/' + vtt.replace('.vtt', '.txt')

                with open(transcript_file, "w") as output_file:
                    output_file.write(video_transcript_2_txt.vtt2text(course_dir + '/English VTT Files/' + vtt))
                    output_file.close()


if __name__ == '__main__':
    parsing_all_csv(var_config.csv_course_dir)
