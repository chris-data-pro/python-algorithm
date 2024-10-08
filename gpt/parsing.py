import csv
import re
from pathlib import Path
from gpt.common import var_config

if __name__ == '__main__':
    csv_file = var_setup.input_file

    data_dir = 'data/' + var_setup.module_name

    # create data path directory if not exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    txt_file = data_dir + '/content.txt'
    # print(txt_file)

    def striphtml(data):
        p = re.compile(r'<.*?>')
        return p.sub('', data.replace('&nbsp;', ' ').replace(' ', '').replace(' ', ''))

    with open(txt_file, "w") as my_output_file:
        with open(csv_file, "r") as my_input_file:
            [my_output_file.write("".join(striphtml(row[1])) + '\n')
             for row in csv.reader(my_input_file)
             if row[0].endswith('/title/') or row[0].endswith('/body/') or row[0].endswith('/text/')]
        my_output_file.close()
    # with open(csv_file, "r") as my_input_file:
    #     print(" ".join(row) for row in csv.reader(my_input_file) if 'title' in row[0])
