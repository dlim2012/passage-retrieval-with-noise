
import random
import os
import csv
from tqdm import tqdm
import string
import sys
from concurrent.futures import ThreadPoolExecutor
import nltk
from inflection import pluralize, singularize

nltk.download('omw-1.4')

# Set the reading and saving directories
read_dir = '/mnt/ssd/data/ms_marco/original'
save_dir = '/mnt/ssd/noisy'
os.makedirs(save_dir, exist_ok=True)

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
    swap word order in each sentence
    swap neighboring word order if the parameter 'adjacent' is 'True'
    """
    sentences = text.split('.')
    for i in range(len(sentences)):
        words = sentences[i].split(' ')
        if len(words) < 2:
            continue
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
    Each word with prob p
    Repeat the process if no word has been attempted to be changed
    """
    words = text.split(' ')

    indices = indices_to_change(words, p)

    for idx in indices:
        word = words[idx]

        if error_type == 'deletion':
            try:
                x = random.randrange(0, len(word))
                word = word[:x] + word[x + 1:]
            except:
                pass
        elif error_type == 'insertion':
            x = random.randrange(0, len(word) + 1)
            character_to_add = random.choice(character_choices)
            word = word[:x] + character_to_add + word[x:]
        elif error_type == 'neighboring_swap':
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
    Repeat the process if no word has been attempted to be changed
    """

    words = text.split(' ')

    indices = indices_to_change(words, p)

    for idx in indices:
        word = words[idx]
        words[idx] = lemmatizer.lemmatize(word)

    return ' '.join(words)


def pluralize_singularize(text, p=0.5):
    """
    With probability of p/2, pluralize each word
    With probability of p/2, singularize each word
    Repeat the process if no word has been attempted to be changed
    """
    words = text.split(' ')
    indices = indices_to_change(words, p)

    for idx in indices:
        word = words[idx]

        if random.random() > 0.5:
            words[idx] = singularize(word)
        else:
            words[idx] = pluralize(word)

    return ' '.join(words)


def remove_space(text, p=0.1):  # 6980
    words = text.split(' ')

    indices = set()
    if p == -1:
        if len(words) > 1:
            indices.add(random.randrange(0, len(words)-1))
    else:
        for idx in range(len(words)-1):
            if random.random() < p:
                indices.add(idx)

    result = ''
    for idx, word in enumerate(words):
        if idx not in indices:
            result += ' '
        result += words[idx]

    return result

def remove_stopwords(text, p=0.2):
    words = text.split(' ')

    stopwords = nltk.corpus.stopwords.words('english')

    indices = []
    for idx, word in enumerate(words):
        if word in stopwords:
            indices.append(idx)

    if p == -1:
        if len(indices) >= 1:
            indices = [random.choice(indices)]
    else:
        indices = [idx for idx in indices if random.random() < p]

    result = [words[idx] for idx in range(len(words)) if idx not in indices]

    return ' '.join(result)


def generate_noise(read_path, save_path, function, args, desc):
    writer = open(save_path, 'w')

    file_len = 0
    with open(read_path, 'r') as file:
        for _ in file:
            file_len += 1

    with open(read_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        pbar = tqdm(enumerate(reader), desc=desc, total=file_len)
        for i, line in pbar:
            text_id, text = line[0], line[1]

            noisy_text = function(text, **args)
            writer.write('%s\t%s\n' % (text_id, noisy_text))



def main():
    noise_functions = [
        ('word_order_swap', passages_filename, word_order_swap, {'adjacent': False}),
        ('word_order_swap_adjacent', queries_dev_filename, word_order_swap, {'adjacent': True}),
        ('neighboring_character_swap_one', queries_dev_filename, character_error, {'error_type': 'neighboring_swap', 'p': -1}),
        ('remove_space_one', queries_dev_filename, remove_space, {'p': -1}),
        ('lemmatize_0.5', queries_dev_filename, lemmatize, {'p': 0.5}),
        ('remove_stopwords_one', queries_dev_filename, remove_stopwords, {'p': -1}),

        ('word_order_swap', queries_dev_filename, word_order_swap, {'adjacent': False}),
        ('word_order_swap_adjacent', passages_filename, word_order_swap, {'adjacent': True}),
        ('neighboring_character_swap_0.1', passages_filename, character_error, {'error_type': 'neighboring_swap', 'p': 0.1}), #v
        ('remove_space_0.1', passages_filename, remove_space, {'p': 0.1}), #v
        ('lemmatize_0.5', passages_filename, lemmatize, {'p': 0.5}), #v
        ('remove_stopwords_0.2', passages_filename, remove_stopwords, {'p': 0.2}),



        ('lemmatize_1', queries_dev_filename, lemmatize, {'p': 1}),
        ('remove_stopwords_0.5', queries_dev_filename, remove_stopwords, {'p': 0.5}),

        ('lemmatize_1', passages_filename, lemmatize, {'p': 1}),
        ('remove_stopwords_0.5', passages_filename, remove_stopwords, {'p': 0.5}),

        # ('pluralize_singularize_0.5', queries_dev_filename, pluralize_singularize, {'p': 0.5}),
        # ('pluralize_singularize_0.5', passages_filename, pluralize_singularize, {'p': 0.5})
    ]

    noise_functions = [
        ('remove_space_one', queries_dev_filename, remove_space, {'p': -1}),
        ('remove_space_0.1', passages_filename, remove_space, {'p': 0.1}), #v
    ]

    threadpool = ThreadPoolExecutor(10)

    for key, filename, function, args in noise_functions:
        read_path = os.path.join(read_dir, '.'.join([filename, 'tsv']))
        write_path = os.path.join(save_dir, '.'.join([filename, key, 'tsv']))

        desc = '.'.join([filename, key])

        #threadpool.submit(generate_noise, read_path, write_path, function, args, desc)
        generate_noise(read_path, write_path, function, args, desc)



if __name__ == '__main__':
    main()
