import os

# Directory for MS Marco dataset
data_dir = '/mnt/ssd/data/ms_marco'

######################## Directories
original_dir = os.path.join(data_dir, 'original')
cleaned_dir = os.path.join(data_dir, 'latin_cleaned')
write_dir = os.path.join(data_dir, 'preprocessed')
reranker_dir = os.path.join(write_dir, 'reranker')
dpr_dir = os.path.join(write_dir, 'dpr')
colbert_dir = os.path.join(write_dir, 'colbert')

######################## Original files
train_query_file_0 = os.path.join(original_dir, 'queries.train.tsv')
dev_query_file_0 = os.path.join(original_dir, 'queries.dev.tsv')
passage_file_0 = os.path.join(original_dir, 'collection.tsv')
triples_train_small_file_0 = os.path.join(original_dir, 'triples.train.small.tsv')

train_qrels_file = os.path.join(original_dir, 'qrels.train.tsv')
dev_qrels_file = os.path.join(original_dir, 'qrels.dev.tsv')
qidpidtriples_file = os.path.join(original_dir, 'qidpidtriples.train.full.tsv')

# big files
#train_top1000_file = os.path.join(read_dir, 'top1000.train.txt')
#dev_top1000_file = os.path.join(read_dir, 'top1000.dev')
train_top1000_file = '/mnt/hdd/MS_Marco/top1000.train.txt'
dev_top1000_file = '/mnt/hdd/MS_Marco/top1000.dev'


######################## Cleaned files
train_query_file = os.path.join(cleaned_dir, 'queries.train.tsv')
dev_query_file = os.path.join(cleaned_dir, 'queries.dev.tsv')
passage_file = os.path.join(cleaned_dir, 'collection.tsv')
triples_train_small_file = os.path.join(cleaned_dir, 'triples.train.small.tsv')



######################## Preprocessed files for general use
# top 1000 pids for each qid saved in npy files
# the first column is qid and others are pid and paddings(-1)
top1000_train_id_file = os.path.join(write_dir, "top1000.train.id.all.npy")
top1000_dev_id_file = os.path.join(write_dir, "top1000.dev.id.all.npy")

# Lines of 'qidpidtriples.train.full.tsv' are shuffled in a tsv file
qidpidtriples_shuffled_file = os.path.join(write_dir, 'qidpidtriples.train.full.shuffled.tsv')
triples_train_small_shuffled_file = os.path.join(write_dir, 'triples.train.small.shuffled.tsv')

# top1000 ids (in each line: the first column is qid and the rest are pids)
train_top1000_id_file = os.path.join(write_dir, 'top1000.train.id.tsv')
train_query_top1000_file = os.path.join(write_dir, 'queries.train.top1000.tsv')
dev_top1000_id_file = os.path.join(write_dir, 'top1000.dev.id.tsv')
dev_query_top1000_file = os.path.join(write_dir, 'queries.dev.top1000.tsv')

# Reranked top1000 using bm25 and all passages in passage_file
train_top1000_id_bm25_file = os.path.join(write_dir, 'top1000.train.id.bm25.tsv')
#train_top1000_id_bm25_npy = os.path.join(write_dir, 'top1000.train.id.bm25.npy')
dev_top1000_id_bm25_file = os.path.join(write_dir, 'top1000.dev.id.bm25.tsv')
#dev_top1000_id_bm25_npy = os.path.join(write_dir, 'top1000.dev.id.bm25.npy')


######################## Reranker data
train_qidpidlabel_file = os.path.join(reranker_dir, 'train.qidpidlabel.tsv')
valid_qidpidlabel_file = os.path.join(reranker_dir, 'validation.qidpidlabel.tsv')




######################## DPR data



######################## ColBERT data
