import os
import re
import csv
import json
import docx2txt
from pathlib import Path
from helper import video_transcript_2_txt
from common import var_config


def parsing_all_json(course_dir, data_dir):
    """Parses Course Text Contents From All Modules in json and Videos in vtt.

    Parameters
    ----------
    course_dir : str, required
        The path of the course.
    data_dir : str, required
        The path of the output txt data.
    """

    course = course_dir.split('/')[-1]

    print('Parsing {} Course Text Contents From All Modules and Videos'.format(course))

    # create data path directory if not exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for module in os.listdir(course_dir):

        if module.startswith('m'):

            json_file_course = course_dir + '/' + module + '/course/en-US/course.json'
            json_file_components = course_dir + '/' + module + '/course/en-US/components.json'
            tincan = course_dir + '/' + module + '/tincan.xml'
            f = open(tincan, "r")
            module_url = (f.read()).split("<activity id=")[1].split(" ")[0]

            txt_file = data_dir + '/' + module + '.txt'
            # print(txt_file)

            def striphtml(data):
                p = re.compile(r'<.*?>')
                return p.sub('', data.replace('&nbsp;', ' ').replace('{{_globals._moduleNumber}}', module[1:]))

            with open(txt_file, "w") as my_output_file:

                if os.path.isfile(json_file_course):
                    with open(json_file_course, "r") as my_title_file:
                        json_object = json.load(my_title_file)
                        if 'title' in json_object:
                            my_output_file.write(
                                striphtml('Module Title: ' + json_object['title']) + '\n' +
                                'Module URL: ' + module_url + '\n'
                            )
                        if 'displayTitle' in json_object:
                            if json_object['displayTitle']:
                                my_output_file.write(striphtml(json_object['displayTitle']) + '\n')
                        if 'body' in json_object:
                            my_output_file.write(striphtml(json_object['body']) + '\n')

                if os.path.isfile(json_file_components):
                    with open(json_file_components, "r") as my_input_file:
                        # [my_output_file.write(striphtml(row['title']) + '\n' +
                        #                       striphtml(row['body']) + '\n' +
                        #                       striphtml(row['_feedback']['correct']) + '\n')
                        #  for row in json.load(my_input_file)]
                        for row in json.load(my_input_file):
                            if 'title' in row:
                                my_output_file.write(striphtml(row['title']) + '\n')
                            if 'displayTitle' in row:
                                if row['displayTitle']:
                                    my_output_file.write(striphtml(row['displayTitle']) + '\n')
                            if 'body' in row:
                                my_output_file.write(striphtml(row['body']) + '\n')
                            # Do not include any quizes or Q&As
                            # if '_feedback' in row:
                            #     if 'correct' in row['_feedback']:
                            #         my_output_file.write(striphtml(row['_feedback']['correct']) + '\n')

                my_output_file.close()

        if module.endswith('VTT Files'):
            for vtt in os.listdir(course_dir + '/' + module):

                transcript_file = data_dir + '/' + vtt.replace('.vtt', '.txt')
                full_name = vtt.replace('.vtt', '')
                video_no = full_name.split(' - ')[0] if ' - ' in full_name else full_name.split(' ')[0]
                module_no = video_no.split('.')[0]
                section_no = '.'.join(video_no.split('.')[:2])

                with open(transcript_file, "w") as output_file:
                    output_file.write(
                        'Module Number: ' + module_no + '\n' +
                        'Section Number: ' + section_no + '\n' +
                        'Video Number: ' + video_no + '\n' +
                        'Video Title: ' + (full_name.split(' - ')[1] if ' - ' in full_name else full_name) + '\n' +
                        'Video URL: ' + '\n' +
                        video_transcript_2_txt.vtt2text(course_dir + '/' + module + '/' + vtt)
                    )
                    output_file.close()

        if module == 'word-docs':
            # print('Parsing word documents')
            for doc in os.listdir(course_dir + '/word-docs'):
                meta_file = data_dir + '/' + doc.replace('.docx', '.txt')

                with open(meta_file, "w") as output_file:
                    output_file.write(docx2txt.process(course_dir + '/word-docs/' + doc).replace(' ', ''))
                    output_file.close()


if __name__ == '__main__':
    course = var_config.course_dir.split('/')[-1]
    parsing_all_json(var_config.course_dir, var_config.cwd + '/data/' + course)
