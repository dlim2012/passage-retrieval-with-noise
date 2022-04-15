"""
1. Split text into multiple small files
2. Use multithread to backtranslate each small file
3. Merge the output and remove the splitted small files
"""

#pip install googletrans==3.1.0a0
from googletrans import Translator, constants
import csv
import os
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse


# Set directory
read_dir = 'data/ms_marco/original'
copy_dir = '/mnt/ssd/tmp'
save_dir = '/mnt/ssd/noisy/backtranslation'
os.makedirs(copy_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--make_initial_copy', default=False, action='store_true')

    return parser.parse_args()

def split_text(read_file, write_file, split_size):
    """
    """
    file_num = -1

    with open(read_file, 'r') as file:
        for i, line in enumerate(file):
            if i % split_size == 0:
                try:
                    writer.close()
                except:
                    pass
                file_num += 1

                ###########################
                #if file_num == 5: break
                ##########################

                writer = open(write_file % file_num, 'w')

            writer.write(line)
    writer.close()

    return file_num

def merge_text(read_filenames, write_filename, dir_path):
    write_file = os.path.join(dir_path, write_filename)
    read_files = [os.path.join(dir_path, read_filename) for read_filename in read_filenames]

    writer = open(write_file, 'w')

    for read_file in read_files:
        with open(read_file, 'r') as file:
            for i, line in enumerate(file):
                writer.write(line)
    writer.close()




def performBackTranslation(inputData):

    # init the Google API translator
    translator = Translator()
    #English to Target Language
    translations = translator.translate(inputData, src='en', dest='fr')
    targetTranslations = []
    for translation in translations:
        targetTranslations.append(translation.text)

    #Translate back from target language to english
    translations = translator.translate(targetTranslations, src='fr', dest='en')
    englishTranslations = []
    for translation in translations:
        englishTranslations.append(translation.text)
    return englishTranslations

def backtranslate(read_path, save_path, batch_size=10, remove_read_path=True):
    """
    Caution: removing read_path after making file in save_path by default
    """

    writer = open(save_path, 'w')

    line_count = 0
    for _ in open(read_path, 'r'):
        line_count += 1

    print('%s > %s, (%d lines)' % (read_path, save_path, line_count))
    ####################
    #line_count = 100
    ####################

    with open(read_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')

        batch_ids, batch_texts = [], []
        for i, line in tqdm(enumerate(reader)):
            ####################
            #if i == 100: break
            ####################

            batch_ids.append(line[0])
            batch_texts.append(line[1])
            if (i+1) % batch_size == 0 or (i+1) == line_count - 1:
                batch_texts = performBackTranslation(batch_texts)

                for j in range(len(batch_ids)):
                    writer.write('%s\t%s\n' % (batch_ids[j], batch_texts[j]))

                batch_ids, batch_texts = [], []

    writer.close()

    if remove_read_path:
        os.remove(read_path)

def main():
    args = parse()
    print('make_initial_copy:', args.make_initial_copy)

    targetLanguage = "fr"
    batch_size = 10

    # directories

    # file names
    # queries_train_filename = 'original/queries.train'
    #queries_dev_filename = 'queries.dev'
    #passages_filename = 'collection'

    os.makedirs(save_dir, exist_ok=True)

    query_dev_filename = 'queries.dev.tsv'
    query_dev_save_filename = 'queries.dev.backtranslation.tsv'
    query_dev_file = os.path.join(read_dir, 'queries.dev.tsv')
    query_dev_copy_file = os.path.join(copy_dir, 'queries.dev.tsv')

    passage_filename = 'collection.tsv'
    passage_split_filename = 'collection_split%d.tsv'
    passage_save_filename = 'collection.backtranslation.tsv'
    passage_file = os.path.join(read_dir, 'collection.tsv')
    split_file = os.path.join(copy_dir, 'collection_split%d.tsv')
    split_size = 10000

    if args.make_initial_copy:
        print('copying')
        shutil.copyfile(query_dev_file, query_dev_copy_file)
        file_num = split_text(passage_file, split_file, split_size); print(file_num)
    else:
        file_num = 884

    filenames =  [passage_split_filename % i for i in range(file_num)]

    # backtranslate dev queries
    t = threading.Thread(
        target=backtranslate,
        args=(os.path.join(copy_dir, query_dev_save_filename), os.path.join(save_dir, query_dev_filename))
    )
    t.start()

    # backtranslate splitted collection files with a thread pool
    with ThreadPoolExecutor(max_workers=20) as executor:
        for i, filename in enumerate(filenames):
            read_path = os.path.join(copy_dir, filename)
            write_path = os.path.join(save_dir, filename)
            if not os.path.exists(read_path):
                continue
            executor.submit(backtranslate, read_path, write_path)

    # merge texts
    if len(os.listdir(copy_dir)) == 0:
        split_filenames = [passage_split_filename % i for i in range(file_num)]
        merge_text(split_filenames, passage_save_filename, save_dir)




if __name__ == '__main__':
    main()
