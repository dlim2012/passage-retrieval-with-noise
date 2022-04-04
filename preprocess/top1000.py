## Make top1000.____.id.npy

import os
from collections import defaultdict
import numpy as np

read_dir = 'data/ms_marco/original'
write_dir = 'data/ms_marco/preprocessed'
    
read_files = ['top1000.train.txt', 'top1000.dev']#, 'top1000.eval']
write_files = ['top1000.train.id.npy', 'top1000.dev.id.npy']#, 'top1000.eval.id.npy']

use_only_len_1000 = False
if use_only_len_1000:
    write_files = ['top1000.train.id.npy', 'top1000.dev.id.npy']#, 'top1000.eval.id.npy']
else:
    write_files = ['top1000.train.id.all.npy', 'top1000.dev.id.all.npy']#, 'top1000.eval.id.npy']



if __name__ == '__main__':

    print("use_only_len_1000:", use_only_len_1000, '\n')

    # Make directory to save npy files if not exist  
    os.makedirs(write_dir, exist_ok=True)

    for i in reversed(range(2)):
        read_path = os.path.join(read_dir, read_files[i])
        write_path = os.path.join(write_dir, write_files[i])
    
        # Print file names
        print("read_path:", read_path)
        print("write_path:", write_path)
    
        # Declare a dictionary (key: query_id, value: passage_ids in list)
        to_save = defaultdict(lambda: [])

        # Open file
        with open(read_path, 'r') as file:

            # Read each line and save label for each query_id in to_save
            for i, line in enumerate(file):
                query_id, passage_id = map(int, line.split('\t')[:2])
                to_save[query_id].append(passage_id)


        # Print total number of query_id
        print('len(to_save.keys()):', len(to_save.keys()))

        # Calculate the maximum length of labels and count how many have exactly 1000 labels
        max_len = -1
        count = 0
        count_1000 = 0
        for key in to_save.keys():
            max_len = max(max_len, len(to_save[key]))
            if len(to_save[key]) == 1000:
                count_1000 += 1

        # Print maximum labels
        print('max_len:', max_len)

        # Print how many query_ids have exactly 1000 labels
        print('count_1000:', count_1000)
    
        # Make a numpy array where the first column is the query_id and the rest columns are passage_ids
        # Fill blanks with -1
        array = []
        if use_only_len_1000:
            for key in to_save.keys():
                if len(to_save[key]) != 1000:
                    continue
                array.append([key] + to_save[key])
        else:
            for key in to_save.keys():
                array.append([key] + to_save[key] + [-1] * (max_len - len(to_save[key])))
        array = np.array(array, dtype=np.int32)
    
        # Print some information about array (np.int32: -2_147_483_648 to 2_147_483_647)
        print(array.shape, array.dtype, array.max())

        # Save the numpy array
        np.save(write_path, array)
    
        print()
    
    
