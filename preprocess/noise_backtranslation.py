"""
1. Split text into multiple small files
2. Use multithread to backtranslate each small file
3. Merge the output and remove the splitted small files
"""

#pip install googletrans==3.1.0a0
from googletrans import Translator, constants
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse

from tools.paths import *

# Set directory
copy_dir = os.path.join(write_dir, 'noisy/backtranslation_split')
split_save_dir = os.path.join(write_dir, 'noisy/backtranslation')
save_dir = os.path.join(write_dir, 'noisy')

os.makedirs(copy_dir, exist_ok=True)
os.makedirs(split_save_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)


def parse():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--make_initial_copy', default=False, action='store_true')
    parser.add_argument('--target_language', type=str, default='fr')
    parser.add_argument('--split_size', type=int, default=10000)

    # file_num: any number bigger than the maximum file number used in split files
    parser.add_argument('--file_num', type=int, default=10000)

    return parser.parse_args()


def split_text(read_file, write_file, split_size):
    """
    Split a large text file into multiple small text file
    This is used to prevent restarting the whole process when connection to Google translation is disconnected
    :param read_file: file to read from
    :param write_file: file to write
    :param split_size: lines of each split file
    :return: the last file number
    """
    file_num = -1

    # Open read file
    with open(read_file, 'r') as file:
        for i, line in enumerate(file):
            # Close the previous write file and open a new write file at each split_size
            if i % split_size == 0:
                try:
                    writer.close()
                except:
                    pass
                file_num += 1

                writer = open(write_file % file_num, 'w')

            # Write a line from the read file
            writer.write(line)

    # Close write file
    writer.close()

    # Return the final file number
    return file_num


def merge_text(read_files, write_file, split_size):
    """
    Merge multiple files into one file
    :param read_files: read files
    :param write_filename: write file
    :param split_size:
    :return:
    """

    # Open write file
    writer = open(write_file, 'w')

    for read_file in read_files:

        # Skip not existing read files
        if not os.path.exists(read_file):
            continue

        # Read a read file and write to the write file
        with open(read_file, 'r') as file:
            i = 0
            for i, line in enumerate(file):
                writer.write(line)

            # Print a message when the number of lines in a read_file is not equal to split_size
            if i != split_size - 1:
                print('Warning: %s has only %d lines' % (read_file, i))

    # Close write file
    writer.close()



def performBackTranslation(inputData, target_language):
    """
    Perform backtranslation
    :param inputData: a list of English texts
    :param target_language: a valid target language for googletrans (e.g. 'fr': french, 'cs': czech, 'zh-cn': chinese)
    :return: backtranslation results
    """

    # init the Google API translator
    translator = Translator()

    # Translate English to target Language
    translations = translator.translate(inputData, src='en', dest=target_language)
    targetTranslations = []
    for translation in translations:
        targetTranslations.append(translation.text)

    #Translate back from target language to English
    translations = translator.translate(targetTranslations, src=target_language, dest='en')
    englishTranslations = []
    for translation in translations:
        englishTranslations.append(translation.text)

    # Return results
    return englishTranslations

def backtranslate(read_path, save_path, target_language='fr', desc='', batch_size=10, remove_read_path=True):
    """
    Caution: removing read_path after making file in save_path by default
    """

    # Open write file
    writer = open(save_path, 'w')

    # Count number of lines in read_path to use in tqdm
    line_count = 0
    for _ in open(read_path, 'r'):
        line_count += 1

    # Print out the operation it started
    print('%s > %s, (%d lines)' % (read_path, save_path, line_count))

    # Open read file
    with open(read_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')

        # Make batches and perform Backtranslation through Google's service to each batch
        batch_ids, batch_texts = [], []
        for i, line in tqdm(enumerate(reader), desc=desc, total=line_count):

            # Make batch
            batch_ids.append(line[0])
            batch_texts.append(line[1])

            # Perform google trans
            if (i+1) % batch_size == 0 or (i+1) == line_count - 1:
                batch_texts = performBackTranslation(batch_texts, target_language)

                # Remove any line split character('\n') generated during backtranslation
                for i, text in enumerate(batch_texts):
                    if '\n' in text:
                        batch_texts[i] = text.replace('\n', ' ')

                # Save results in write file
                for j in range(len(batch_ids)):
                    writer.write('%s\t%s\n' % (batch_ids[j], batch_texts[j]))

                # Reset batch
                batch_ids, batch_texts = [], []

    # Close writer
    writer.close()

    # Remove read file if finished
    if remove_read_path:
        os.remove(read_path)

def main():
    args = parse()
    print('make_initial_copy:', args.make_initial_copy)

    os.makedirs(save_dir, exist_ok=True)

    # Path to save split files for dev queries and passages
    query_split_filename = 'queries.dev.split%d_' + '%s.tsv' % args.target_language
    query_split_file = os.path.join(copy_dir, query_split_filename)
    passage_split_filename = 'collection_split%d_' + '%s.tsv' % args.target_language
    passage_split_file = os.path.join(copy_dir, passage_split_filename)

    # Path to save split final results for queries and passages
    query_dev_save_file = os.path.join(save_dir, 'queries.dev.backtranslation_%s.tsv' % args.target_language)
    passage_save_file = os.path.join(save_dir, 'collection.backtranslation_%s.tsv' % args.target_language)


    # If make_initial_copy flag is used, make split files and exit program
    if args.make_initial_copy:
        print('copying...')
        split_text(passage_file, passage_split_file, args.split_size)
        split_text(query_dev_file, query_split_file, args.split_size)
        import sys; sys.exit(0)


    # Backtranslate splitted collection files with a thread pool
    # Connection to googletrans could fail for many split files
    # -> Run this program multiple times until all split read files are deleted after backtranslation
    with ThreadPoolExecutor(max_workers=20) as executor:
        for filename_ in [query_split_filename, passage_split_filename]:
            for i in range(args.file_num):
                filename = filename_ % i
                read_path = os.path.join(copy_dir, filename)
                write_path = os.path.join(split_save_dir, filename)
                if not os.path.exists(read_path):
                    continue
                executor.submit(backtranslate, read_path, write_path, args.target_language, str(i).rjust(3, ' '))

    # Merge texts if all split read files are backtranslated
    if len(os.listdir(copy_dir)) == 0:
        # Merge passage file splits
        passage_split_files = [os.path.join(split_save_dir, passage_split_filename % i) for i in range(args.file_num)]
        merge_text(passage_split_files, passage_save_file, args.split_size)

        # Merge query dev file splits
        query_split_filenames = [os.path.join(split_save_dir, query_split_filename) % i for i in range(args.file_num)]
        merge_text(query_split_filenames, query_dev_save_file, args.split_size)

if __name__ == '__main__':
    main()
