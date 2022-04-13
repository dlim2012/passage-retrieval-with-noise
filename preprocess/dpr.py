
# import required packages
import csv
import os

# import from other files
from tools.paths import *
from tools.read_files import read_texts
from tools.tools import shuffle_text

# save directory and the number of pairs to save
save_dir = os.path.join(dpr_dir, 'train')
save_path = os.path.join(save_dir, 'qidpidtriples.train.full.filtered.text.tsv')
max_size = 200000 * 32


def preprocess_train():
    """
    preprocess train data for DPR
        1. Shuffled data: qidpidtriples_file > qidpidtriples_shuffled_file
        2. Filter qidpidtriples_shuffled_file so that negative passages are not relevant to any train query
        3. convert ids to text and save to 'qidpidtriples.train.full.filtered.text.tsv'
    """
    # Make save directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    # 1. shuffle lines in the qidpidtriples file downloaded from the official website
    if not os.path.exists(qidpidtriples_shuffled_file):
        shuffle_text(qidpidtriples_file, qidpidtriples_shuffled_file)


    rel_ids = set()
    with open(train_qrels_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            rel_ids.add(int(line[2]))

    # Read queries and passages
    queries = read_texts(train_query_file)
    passages = read_texts(passage_file)

    # Open write file
    write_file = open(save_path, 'w')

    # Write lines until the number of lines reaches the target number
    count = 0
    with open(qidpidtriples_shuffled_file, 'r') as read_file:
        reader = csv.reader(read_file, delimiter='\t')
        for i, line in enumerate(reader):
            negative_pid = int(line[2])

            # Filter negative passages that are relevant to any query
            # This is to use the negative passages as "hard" negatives without being relevant passages
            if negative_pid in rel_ids:
                continue

            # Write one line
            line = '\t'.join([queries[int(line[0])], passages[int(line[1])], passages[int(line[2])]]) + '\n'
            write_file.write(line)
            count += 1

            if count == max_size:
                break

    write_file.close()


if __name__ == '__main__':
    
    # preprocess train data in format 'query, tab, positive passage, tab, negative passage)
    preprocess_train()

