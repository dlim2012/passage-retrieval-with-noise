#Load Data and Packages
from textattack.transformations import WordSwapRandomCharacterInsertion
from textattack.transformations import WordSwapRandomCharacterDeletion
from textattack.transformations import WordSwapRandomCharacterSubstitution
from textattack.transformations import WordSwapNeighboringCharacterSwap
#from textattack.transformations import WordSwapQWERTY
#from textattack.transformations import WordSwapChangeLocation
from textattack.transformations import WordSwapInflections
#from textattack.transformations.sentence_transformations import BackTranslation
from textattack.augmentation import Augmenter
import pandas as pd
import random
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

noiseLevel = 0.1 # Change amount of noise generated here (Percentage of words in input, not chars)

# Set the reading and saving directories
read_dir = 'data/ms_marco/original'
save_dir = '/mnt/ssd/noisy'
os.makedirs(save_dir, exist_ok=True)

# file names to use: ('%s.tsv' % file_name)
queries_dev_filename = 'queries.dev'
passages_filename = 'collection'

# read data
queries_dev = pd.read_csv(os.path.join(read_dir, '%s.tsv' % queries_dev_filename), sep='\t', header = None)
collection = pd.read_csv(os.path.join(read_dir, '%s.tsv' % passages_filename), sep='\t', header = None)

def word_order_swap(text, adjacent=False):
    """
    swap word order
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


def generateNoise(outputfileName, inputData, transformationType, desc):
    """
    generate noise
    """
    print('###### Start: %s #######' % outputfileName)
    augmenter = Augmenter(transformation=transformationType, pct_words_to_swap=noiseLevel)
    for x in tqdm(range(0, inputData.shape[0]), desc=desc):
        #print(inputData.at[x, 1])
        if transformationType in [word_order_swap]:
            inputData.at[x,1] = word_order_swap(inputData.at[x, 1])
        else:
            inputData.at[x,1] = (augmenter.augment(inputData.at[x,1]))[0]
    inputData.to_csv(outputfileName, sep = '\t', header = None, index = False)
    print('###### Done: %s #######' % outputfileName)

# file names and readers
files = {
    queries_dev_filename: queries_dev,
    passages_filename: collection
}

# transformation types
transformations = {
    'WordSwapInflections': WordSwapInflections(),
    'WordSwapRandomCharacterInsertion': WordSwapRandomCharacterInsertion(),
    'WordSwapRandomCharacterDeletion': WordSwapRandomCharacterDeletion(),
    'WordSwapRandomCharacterSubstitution': WordSwapRandomCharacterSubstitution(),
    'WordSwapNeighboringCharacterSwap': WordSwapNeighboringCharacterSwap(),
    'word_order_swap': word_order_swap
}

def main():

    # Generate noise using a process pool.
    # (Somehow this doesn't use most of available CPUs. Perhaps should use Multiprocessing.Pool instead)
    with ProcessPoolExecutor(10) as executor:
        for filename in files.keys():
            for transformationType in transformations.keys():
                save_path = os.path.join(save_dir, '%s.%s_%.2f.tsv' % (filename, transformationType, noiseLevel))
                executor.submit(generateNoise,
                                save_path,
                                files[filename].copy(),
                                transformations[transformationType],
                                transformationType)

if __name__ == '__main__':
    main()
