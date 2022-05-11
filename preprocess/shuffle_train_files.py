from tools.tools import shuffle_text

from tools.paths import *

shuffle_text(triples_train_small_file, triples_train_small_shuffled_file)
shuffle_text(qidpidtriples_file, qidpidtriples_shuffled_file)