import numpy as np
import random

def one_hot_vector(x, n_values=None):
    """
    :param x: one dimensional list or np.ndarray
    return one hot vector
    """
    x = np.array(x)
    if not n_values:
        n_values = np.max(x) + 1
    return np.eye(n_values)[x]

def shuffle_text(read_path, write_path):
    """
    Shuffle lines in read_path and write to write_path
    """
    with open(read_path, 'r') as f:
        lines = f.readlines()
            
    random.shuffle(lines)

    with open(write_path, 'w') as f:
        f.writelines(lines)
