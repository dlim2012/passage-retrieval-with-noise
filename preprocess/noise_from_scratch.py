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
character_choices += [' ', '\t', '\n'] # some white spaces

lemmatizer = nltk.stem.WordNetLemmatizer()

def word_order_swap(text, adjacent=False):
    """
    swap word order
    swap neighboring word order if the parameter 'adjacent' is 'True'
    """
    sentences = text.split('.')
    for i in range(len(sentences)):
        words = sentences[i].split(' ')
        if len(words) < 2:
            continue
        if adjacent:
            idx1 = random.choice([_ for _ in range(len(words)-1)])
            idx2 = idx1 + 1
        else:
            idx1, idx2 = random.sample([_ for _ in range(len(words))], k=2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        sentences[i] = ' '.join(words)
    return '.'.join(sentences)

def one_character_error(text, error_type):
    words = text.split(' ')

    # Get the maximum word length
    max_word_length = max([len(word) for word in words])
    if max_word_length < 3:
        print(' '.join(words))

    keyword_indices = []
    for i, word in enumerate(words):
        # If the maximum word length is bigger than 3, choose a word from words with length bigger than 3
        if max_word_length > 3:
            if len(word) > 3:
                keyword_indices.append(i)
        # If the maximum word length is not bigger than 3, choose a word with the maximum word length
        else:
            if len(word) == max_word_length:
                keyword_indices.append(i)

    index = random.choice(keyword_indices)
    word_to_change = words[index]

    if error_type == 'deletion':
        x = random.randrange(0, len(word_to_change))
        word_to_change = word_to_change[:x] + word_to_change[x+1:]
    elif error_type == 'insertion':
        x = random.randrange(0, len(word_to_change)+1)
        character_to_add = random.choice(character_choices)
        word_to_change = word_to_change[:x] + character_to_add + word_to_change[x:]
    elif error_type == 'neighboring_swap':
        try:
            x = random.randrange(0, len(word_to_change)-1)
            word_to_change = word_to_change[:x] + word_to_change[x+1] + word_to_change[x] + word_to_change[x+2:]
        except:
            pass
    else:
        print('%s is not a valid parameter for the character_error function.')
        sys.exit(1)

    words[index] = word_to_change
    return ' '.join(words)

def character_error(text, error_type, p=0.1):
    """
    Each word with prob p
    Repeat the process if no word has been attempted to be changed
    """
    words = text.split(' ')

    n_changed = 0
    while n_changed == 0:
        for i, word in enumerate(words):
            if random.random() > p:
                continue

            if error_type == 'deletion':
                try:
                    x = random.randrange(0, len(word))
                    word = word[:x] + word[x+1:]
                except:
                    pass
            elif error_type == 'insertion':
                x = random.randrange(0, len(word)+1)
                character_to_add = random.choice(character_choices)
                word = word[:x] + character_to_add + word[x:]
            elif error_type == 'neighboring_swap':
                try:
                    x = random.randrange(0, len(word)-1)
                    word = word[:x] + word[x+1] + word[x] + word[x+2:]
                except:
                    pass
            else:
                print('%s is not a valid parameter for the character_error function.')
                sys.exit(1)

            words[i] = word
            n_changed += 1

    return ' '.join(words)

def lemmatize(text, p=0.5):
    """
    Lemmatize each word with probability of p
    Repeat the process if no word has been attempted to be changed
    """

    words = text.split(' ')

    n_changed = 0
    while n_changed == 0:
        for i, word in enumerate(words):
            if random.random() > p:
                continue

            words[i] = lemmatizer.lemmatize(word)

            n_changed += 1

    return ' '.join(words)

def pluralize_singlurzie(text, p=0.5):
    """
    With probability of (1-p)/2, pluralize each word
    With probability of (1-p)/2, singularize each word
    Repeat the process if no word has been attempted to be changed
    """
    words = text.split(' ')

    n_changed = 0
    while n_changed == 0:
        for i, word in enumerate(words):
            if random.random() > p:
                continue

            if random.random() > 0.5:
                words[i] = singularize(word)
            else:
                words[i] = pluralize(word)
            n_changed += 1

    return ' '.join(words)


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


noise_functions = {
    'word_order_swap': (word_order_swap, {'adjacent': False}),
    'word_adjacent_swap': (word_order_swap, {'adjacent': True}),
    'character_deletion': (character_error, {'error_type': 'deletion'}),
    'character_insertion': (character_error, {'error_type': 'insertion'}),
    'neighboring_character_swap': (character_error, {'error_type': 'neighboring_swap'}),
    'lemmatize_0.5': (lemmatize, {'p': 0.5}),
    'pluralize_singlurize_0.5': (pluralize_singlurzie, {'p': 0.5})
}


# 'one_character_deletion': (one_character_error, {'error_type': 'deletion'}),
# 'one_character_insertion': (one_character_error, {'error_type': 'insertion'}),
# 'one_neighboring_character_swap': (one_character_error, {'error_type': 'neighboring_swap'}),

def main():
    filenames = [queries_dev_filename, passages_filename]
    threadpool = ThreadPoolExecutor(5)

    for filename in filenames:
        read_path = os.path.join(read_dir, '.'.join([filename, 'tsv']))
        for key, value in noise_functions.items():
            write_path = os.path.join(save_dir, '.'.join([filename, key, 'tsv']))
            function, args = value
            desc = '.'.join([filename, key])

            threadpool.submit(generate_noise, read_path, write_path, function, args, desc)
            #generate_noise(read_path, write_path, function, args, desc)

if __name__ == '__main__':
    main()
