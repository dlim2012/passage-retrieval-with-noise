import os
import csv

from tools.paths import *


LATIN_1_CHARS = (
    ('\xe2\x80\x99', "'"),
    ('\xc3\xa9', 'e'),
    ('\xe2\x80\x90', '-'),
    ('\xe2\x80\x91', '-'),
    ('\xe2\x80\x92', '-'),
    ('\xe2\x80\x93', '-'),
    ('\xe2\x80\x94', '-'),
    ('\xe2\x80\x94', '-'),
    ('\xe2\x80\x98', "'"),
    ('\xe2\x80\x9b', "'"),
    ('\xe2\x80\x9c', '"'),
    ('\xe2\x80\x9c', '"'),
    ('\xe2\x80\x9d', '"'),
    ('\xe2\x80\x9e', '"'),
    ('\xe2\x80\x9f', '"'),
    ('\xe2\x80\xa6', '...'),
    ('\xe2\x80\xb2', "'"),
    ('\xe2\x80\xb3', "'"),
    ('\xe2\x80\xb4', "'"),
    ('\xe2\x80\xb5', "'"),
    ('\xe2\x80\xb6', "'"),
    ('\xe2\x80\xb7', "'"),
    ('\xe2\x81\xba', "+"),
    ('\xe2\x81\xbb', "-"),
    ('\xe2\x81\xbc', "="),
    ('\xe2\x81\xbd', "("),
    ('\xe2\x81\xbe', ")")
)


NEW_REPLACEMENT = (
    ('\xe2\x80\x80', ' '),
    ('\xe2\x80\x81', ' '),
    ('\xe2\x80\x82', ' '),
    ('\xe2\x80\x83', ' '),
    ('\xe2\x80\x84', ' '),
    ('\xe2\x80\x85', ' '),
    ('\xe2\x80\x86', ' '),
    ('\xe2\x80\x87', ' '),
    ('\xe2\x80\x88', ' '),
    ('\xe2\x80\x89', ' '),
    ('\xe2\x80\x8a', ' '),
    ('\xe2\x80\x8b', ' '),
    ('\xe2\x80\x8c', ' '),
    ('\xe2\x80\x8d', ' '),
    ('\xe2\x80\x8e', ' '),
    ('\xe2\x80\x8f', ' '),
)


def clean_latin(data):
    for _hex, _char in LATIN_1_CHARS:
        data = data.replace(_hex, _char)
    for _hex, _char in NEW_REPLACEMENT:
        data = data.replace(_hex, _char)
    return data

data = [
    [train_query_file_0, train_query_file],
    [dev_query_file_0, dev_query_file],
    [passage_file_0, passage_file] ,
    [triples_train_small_file_0, triples_train_small_file]
]


def main():
    os.makedirs(read_dir, exist_ok=True)
    for read_path, write_path in data:
        print('read_path:', read_path)
        print('write_path:', write_path)

        writer = open(write_path, 'w')

        with open(read_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, line in enumerate(reader):
                n_text = len(line)
                for j, text in enumerate(line):
                    line[j] = clean_latin(text)

                to_write = '\t'.join(line) + '\n'

                try:
                    assert to_write.count('\t') == n_text - 1
                    assert to_write.count('\n') == 1
                except:
                    print(len(line), to_write.count('\t'), to_write.count('\n'), line)

                writer.write('\t'.join(line) + '\n')

        writer.close()

if __name__ == '__main__':
    main()
