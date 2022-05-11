import random

def shuffle_text(read_path, write_path):
    """
    Shuffle lines in read_path and write to write_path
    """
    with open(read_path, 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)

    with open(write_path, 'w') as f:
        f.writelines(lines)