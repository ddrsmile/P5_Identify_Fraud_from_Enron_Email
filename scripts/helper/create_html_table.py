# -*- coding: utf-8 -*-
import os
import sys

BASE_PATH = os.path.dirname(os.path.abspath(__file__)) 

def create_html_table():
    input_file = sys.argv[1]
    if "../" in input_file:
        input_file = os.path.abspath(os.path.join(BASE_PATH, input_file))
    else:
        input_file = os.path.join(BASE_PATH, input_file)
    
    with open(input_file, 'r') as in_file:
        in_file.readline()
        for line in in_file:
            line = line.strip().split()
            #print line
            html_tag="<td>{0}</td>"
            print "<tr>"
            for i in range(1, len(line)):
                print html_tag.format(line[i]).strip()
            print "</tr>"



if __name__ == "__main__":
    create_html_table()
