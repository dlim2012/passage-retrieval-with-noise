
import random
import os
import csv
from tqdm import tqdm
import string
import sys
from concurrent.futures import ThreadPoolExecutor
import nltk

from tools.paths import *

nltk.download('omw-1.4')

# Set the reading and saving directories

queries_dev_filename = 'queries.dev'
passages_filename = 'collection'

character_choices = []
character_choices += string.ascii_lowercase
character_choices += string.ascii_uppercase
character_choices += string.digits
character_choices += string.punctuation
character_choices += [' ', '\t', '\n']  # some white spaces

lemmatizer = nltk.stem.WordNetLemmatizer()


def indices_to_change(words, p):
    """
    A function used to get randomly select indices of words with probability p
    However, if p is -1, choose one word that has length bigger than 3.
    If p is -1 and the maximum length of word is not bigger than 3, choose one word with the maximum length
    :param words: list of words
    :param p: probability p
    :return: chosen indices
    """
    indices = []
    if p == -1:
        max_word_length = max([len(word) for word in words])
        candidates = []
        if max_word_length > 3:
            for i, word in enumerate(words):
                if len(word) > 3:
                    candidates.append(i)
        else:
            for i, word in enumerate(words):
                if len(word) == max_word_length:
                    candidates.append(i)
        indices.append(random.choice(candidates))
    else:
        for i in range(len(words)):
            if random.random() < p:
                indices.append(i)
    return indices

def word_order_swap(text, adjacent=False):
    """
    swap one pair of words in each sentence
    swap neighboring word order if the parameter 'adjacent' is 'True'
    """

    # Split text to sentences based on '.'
    sentences = text.split('.')
    for i in range(len(sentences)):
        words = sentences[i].split(' ')

        # Don't swap when the number of words is less than 2 in a sentence
        if len(words) < 2:
            continue

        # Swap one pair of words
        if adjacent:
            idx1 = random.choice([_ for _ in range(len(words) - 1)])
            idx2 = idx1 + 1
        else:
            idx1, idx2 = random.sample([_ for _ in range(len(words))], k=2)
        words[idx1], words[idx2] = words[idx2], words[idx1]

        sentences[i] = ' '.join(words)
    return '.'.join(sentences)


def character_error(text, error_type, p=0.1):
    """
    Make character errors: deletion, insertion, swap neighboring characters
    """
    words = text.split(' ')

    # Get the indices of words to change
    indices = indices_to_change(words, p)

    for idx in indices:
        word = words[idx]

        if error_type == 'deletion':

            # Randomly delete one character in chosen words
            try:
                x = random.randrange(0, len(word))
                word = word[:x] + word[x + 1:]
            except:
                pass
        elif error_type == 'insertion':

            # Randomly insert one character to chosen words
            x = random.randrange(0, len(word) + 1)
            character_to_add = random.choice(character_choices)
            word = word[:x] + character_to_add + word[x:]
        elif error_type == 'neighboring_swap':

            # Randomly swap neighboring characters in chosen words
            try:
                x = random.randrange(0, len(word) - 1)
                word = word[:x] + word[x + 1] + word[x] + word[x + 2:]
            except:
                pass
        else:
            print('%s is not a valid parameter for the character_error function.')
            sys.exit(1)

        words[idx] = word

    return ' '.join(words)

def lemmatize(text, p=0.5):
    """
    Lemmatize each word with probability of p
    """

    words = text.split(' ')

    # Randomly choose the indices of words to lemmatize using the function 'indices_to_change'
    indices = indices_to_change(words, p)

    # Lemmatize the selected words
    for idx in indices:
        word = words[idx]
        words[idx] = lemmatizer.lemmatize(word)

    return ' '.join(words)


def remove_space(text, p=0.1):
    """
    Remove each space with probability of p
    """
    # Split text into words based on the space character
    words = text.split(' ')

    # Randomly select the space characters to remove
    # If the parameter p is -1, choose one space character if there is at least one space character
    indices = set()
    if p == -1:
        if len(words) > 1:
            indices.add(random.randrange(0, len(words)-1))
    else:
        for idx in range(len(words)-1):
            if random.random() < p:
                indices.add(idx)

    # Remove the selected space characters
    result = ''
    for idx, word in enumerate(words):
        if idx not in indices:
            result += ' '
        result += words[idx]

    return result

def remove_stopwords(text, p=0.2):
    """
    Remove stopwords with probability of p
    """
    # Split text into words
    words = text.split(' ')

    # Get common stopwords from the nltk package
    stopwords = nltk.corpus.stopwords.words('english')

    # Get the indices of stop words in the list of words
    indices = []
    for idx, word in enumerate(words):
        if word in stopwords:
            indices.append(idx)

    # Randomly remove each stopword with probability of p
    # If p is -1, remove only one stopword if there is at least one stopwords
    if p == -1:
        if len(indices) >= 1:
            indices = [random.choice(indices)]
    else:
        indices = [idx for idx in indices if random.random() < p]

    result = [words[idx] for idx in range(len(words)) if idx not in indices]

    return ' '.join(result)


def generate_noise(read_path, save_path, function, args, desc):
    """
    A function to generate noise
    :param read_path: path to the text file to read
    :param save_path: path to the text file to save noisy texts
    :param function: function to use to generate noise
    :param args: arguments for the function
    :param desc: description of the process that will be shown in tqdm
    """
    writer = open(save_path, 'w')

    # Count the number of lines in the read file
    file_len = 0
    with open(read_path, 'r') as file:
        for _ in file:
            file_len += 1

    # Read from read_path and convert one line at a time
    # Write the results in write_path
    with open(read_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        pbar = tqdm(enumerate(reader), desc=desc, total=file_len)
        for i, line in pbar:
            text_id, text = line[0], line[1]

            noisy_text = function(text, **args)
            writer.write('%s\t%s\n' % (text_id, noisy_text))



def main():

    # Tuples of (Noise type, file name, noise generating function, arguments for the noise generating function)
    noise_functions = [
        ('word_order_swap', queries_dev_filename, word_order_swap, {'adjacent': False}),
        ('word_order_swap_adjacent', queries_dev_filename, word_order_swap, {'adjacent': True}),
        ('neighboring_character_swap_one', queries_dev_filename, character_error, {'error_type': 'neighboring_swap', 'p': -1}),
        ('remove_space_one', queries_dev_filename, remove_space, {'p': -1}),
        ('lemmatize_0.5', queries_dev_filename, lemmatize, {'p': 0.5}),
        ('lemmatize_1', queries_dev_filename, lemmatize, {'p': 1}),
        ('remove_stopwords_one', queries_dev_filename, remove_stopwords, {'p': -1}),
        ('remove_stopwords_0.5', queries_dev_filename, remove_stopwords, {'p': 0.5}),

        ('word_order_swap', passages_filename, word_order_swap, {'adjacent': False}),
        ('word_order_swap_adjacent', passages_filename, word_order_swap, {'adjacent': True}),
        ('neighboring_character_swap_0.1', passages_filename, character_error, {'error_type': 'neighboring_swap', 'p': 0.1}),
        ('remove_space_0.1', passages_filename, remove_space, {'p': 0.1}),
        ('lemmatize_0.5', passages_filename, lemmatize, {'p': 0.5}),
        ('lemmatize_1', passages_filename, lemmatize, {'p': 1}),
        ('remove_stopwords_0.2', passages_filename, remove_stopwords, {'p': 0.2}),
        ('remove_stopwords_0.5', passages_filename, remove_stopwords, {'p': 0.5}),


    ]

    # Use a threadpool and repeat generating noise for all tuples in noise_functions
    threadpool = ThreadPoolExecutor(10)
    for key, filename, function, args in noise_functions:
        # Get the read and write paths
        read_path = os.path.join(read_dir, '.'.join([filename, 'tsv']))
        write_path = os.path.join(noisy_dir, '.'.join([filename, key, 'tsv']))

        # Get the description based on file name and noise type
        desc = '.'.join([filename, key])

        # Submit to the threadpool
        threadpool.submit(generate_noise, read_path, write_path, function, args, desc)



if __name__ == '__main__':
    main()