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

read_dir = 'data/ms_marco/original'

queries_train_filename = 'queries.train'
queries_dev_filename = 'queries.dev'
passages_filename = 'collection'

queries_train = pd.read_csv(os.path.join(read_dir, '%s.tsv' % queries_train_filename), sep='\t', header = None)
queries_dev = pd.read_csv(os.path.join(read_dir, '%s.tsv' % queries_dev_filename), sep='\t', header = None)
collection = pd.read_csv(os.path.join(read_dir, '%s.tsv' % passages_filename), sep='\t', header = None)

#save_dir = 'data/ms_marco/preprocessed/noisy'
save_dir = '/mnt/ssd/noisy'
os.makedirs(save_dir, exist_ok=True)

def word_order_swap(text, adjacent=False):
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


files = {
    queries_train_filename: queries_train,
    queries_dev_filename: queries_dev,
    passages_filename: collection
}


transformations = {
    #'BackTranslation': BackTranslation(), # This takes too long and seem to cause network disconnection
    'WordSwapInflections': WordSwapInflections(), # This works
    #'WordSwapChangeLocation': WordSwapChangeLocation(), # This literally changes location names and not word orders
    'WordSwapRandomCharacterInsertion': WordSwapRandomCharacterInsertion(), # This works
    'WordSwapRandomCharacterDeletion': WordSwapRandomCharacterDeletion(), # This works
    'WordSwapRandomCharacterSubstitution': WordSwapRandomCharacterSubstitution(), # This works
    'WordSwapNeighboringCharacterSwap': WordSwapNeighboringCharacterSwap(), # This works
    #'WordSwapQWERTY': WordSwapQWERTY(), # This gives error (IndexError: Cannot choose from an empty sequence)
    'word_order_swap': word_order_swap
}

transformations_keys = [
    #'WordSwapQWERTY',
    'word_order_swap',
    'WordSwapNeighboringCharacterSwap',
    'WordSwapRandomCharacterSubstitution',
    'WordSwapRandomCharacterDeletion',
    'WordSwapRandomCharacterInsertion',
    #'WordSwapChangeLocation',
    'WordSwapInflections',
    #'BackTranslation'
]

def main():
    files_keys = [queries_dev_filename, passages_filename]#, queries_train_filename]

    transformations_keys = [
        # 'WordSwapQWERTY',
        'word_order_swap',
        'WordSwapNeighboringCharacterSwap',
        'WordSwapRandomCharacterSubstitution',
        'WordSwapRandomCharacterDeletion',
        'WordSwapRandomCharacterInsertion',
        # 'WordSwapChangeLocation',
        'WordSwapInflections',
        # 'BackTranslation'
    ]

    #with ThreadPoolExecutor(max_workers=10) as executor:
    with ProcessPoolExecutor(10) as executor:
        for filename in files_keys:
            for transformationType in transformations_keys:
                save_path = os.path.join(save_dir, '%s.%s_%.2f.tsv' % (filename, transformationType, noiseLevel))
                executor.submit(generateNoise,
                                save_path,
                                files[filename].copy(),
                                transformations[transformationType],
                                transformationType)

if __name__ == '__main__':
    main()
