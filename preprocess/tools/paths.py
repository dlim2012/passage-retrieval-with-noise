import os

# Directory for MS Marco dataset
data_dir = 'data/ms_marco'

# Directory to save checkpoints
ckpt_dir = 'checkpoints'

# Directory to save vectors
vectors_dir = 'vectors'

######################## Directories
read_dir = os.path.join(data_dir, 'original')
write_dir = os.path.join(data_dir, 'preprocessed')
noisy_dir = os.path.join(write_dir, 'noisy')
latin_cleaned_dir = os.path.join(data_dir, 'latin_cleaned')

os.makedirs(write_dir, exist_ok=True)
os.makedirs(noisy_dir, exist_ok=True)

######################## Original files
query_train_file = os.path.join(read_dir, 'queries.train.tsv')
query_dev_file = os.path.join(read_dir, 'queries.dev.tsv')
passage_file = os.path.join(read_dir, 'collection.tsv')
qrels_train_file = os.path.join(read_dir, 'qrels.train.tsv')
qrels_dev_file = os.path.join(read_dir, 'qrels.dev.tsv')
triples_train_small_file = os.path.join(read_dir, 'triples.train.small.tsv')
qidpidtriples_file = os.path.join(read_dir, 'qidpidtriples.train.full.tsv')

# big files
top1000_train_file = os.path.join(read_dir, 'top1000.train.txt')
top1000_dev_file = os.path.join(read_dir, 'top1000.dev')


######################## Preprocessed files for general use

######################## Main train and validation data (Made using 'qidpidtriples_file')
train_data_file = os.path.join(write_dir, 'train.tsv') # train data
validation_data_file = os.path.join(write_dir, 'validation.tsv') # validation data

# top 1000 pids for each qid saved in npy files
# the first column is qid and others are pid and paddings(-1)
#top1000_train_id_file = os.path.join(write_dir, "top1000.train.id.all.npy")
#top1000_dev_id_file = os.path.join(write_dir, "top1000.dev.id.all.npy")

# Lines of 'qidpidtriples.train.full.tsv' are shuffled in a tsv file


######################## Preprocessed top 1000 passages for each query
# top1000 ids
# (in each line: the first column is qid and the rest are pids)
top1000_train_id_file = os.path.join(write_dir, 'top1000.train.id.tsv')
top1000_dev_id_file = os.path.join(write_dir, 'top1000.dev.id.tsv')

# query top1000
# (in each line: qid with top1000 results is in the first column and the corresponding query is in the second column)
query_train_top1000_file = os.path.join(write_dir, 'queries.train.top1000.tsv')
query_dev_top1000_file = os.path.join(write_dir, 'queries.dev.top1000.tsv')

# Reranked top1000 using bm25 and all passages in passage_file
# (format: same as in train_top1000_id_file)
top1000_train_id_bm25_file = os.path.join(write_dir, 'top1000.train.id.bm25.tsv')
top1000_dev_id_bm25_file = os.path.join(write_dir, 'top1000.dev.id.bm25.tsv')

######################## Other training data
qidpidtriples_shuffled_file = os.path.join(write_dir, 'qidpidtriples.train.full.shuffled.tsv')
triples_train_small_shuffled_file = os.path.join(write_dir, 'triples.train.small.shuffled.tsv')
train_top1000_id_bm25_pair_file = os.path.join(write_dir, 'top1000.train.id.bm25.pair_%d.text.tsv')


######################## Latin-Cleaned
query_train_latin_cleaned_file = os.path.join(write_dir, 'queries.train.latin-cleaned.tsv')
query_dev_latin_cleaned_file = os.path.join(read_dir, 'queries.dev.latin-cleaned.tsv')
passage_latin_cleaned_file = os.path.join(read_dir, 'collection.latin-cleaned.tsv')
triples_train_small_latin_cleaned_file = os.path.join(read_dir, 'triples.train.small.latin-cleaned.tsv')
qidpidtriples_latin_cleaned_file = os.path.join(read_dir, 'qidpidtriples.train.full.latin-cleaned.tsv')


######################## noisy file names
noisy_queries_filenames = [
    'queries.dev.backtranslation.tsv',
    'queries.dev.neighboring_character_swap_one.tsv',
    'queries.dev.remove_stopwords_one.tsv',
    'queries.dev.lemmatize_0.5.tsv',
    'queries.dev.remove_space_one.tsv',
    'queries.dev.word_order_swap.tsv',
    'queries.dev.word_order_swap_adjacent.tsv',
    'queries.dev.lemmatize_1.tsv',
    'queries.dev.remove_stopwords_0.5.tsv'
]

noisy_passage_filenames = [
    'collection.backtranslation.tsv',
    'collection.lemmatize_0.5.tsv',
    'collection.lemmatize_1.tsv',
    'collection.neighboring_character_swap_0.1.tsv',
    'collection.remove_space_0.1.tsv',
    'collection.remove_stopwords_0.2.tsv',
    'collection.remove_stopwords_0.5.tsv',
    'collection.word_order_swap.tsv',
    'collection.word_order_swap_adjacent.tsv'
]
