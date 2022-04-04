import os

# Directory for MS Marco dataset
data_dir = 'data/ms_marco'

######################## Directories
read_dir = os.path.join(data_dir, 'original')
write_dir = os.path.join(data_dir, 'preprocessed')
reranker_dir = os.path.join(write_dir, 'reranker')
dpr_dir = os.path.join(write_dir, 'dpr')
colbert_dir = os.path.join(write_dir, 'colbert')

######################## Original files 
train_query_file = os.path.join(read_dir, 'queries.train.tsv')
train_qrels_file = os.path.join(read_dir, 'qrels.train.csv')
dev_query_file = os.path.join(read_dir, 'queries.dev.tsv')
dev_qrels_file = os.path.join(read_dir, 'qrels.dev.tsv')
passage_file = os.path.join(read_dir, 'collection.tsv')
qidpidtriples_file = os.path.join(read_dir, 'qidpidtriples.train.full.tsv')


######################## Preprocessed files for general use
# top 1000 pids for each qid saved in npy files
# the first column is qid and others are pid and paddings(-1)
top1000_train_id_file = os.path.join(write_dir, "top1000.train.id.all.npy")
top1000_dev_id_file = os.path.join(write_dir, "top1000.dev.id.all.npy")

# Lines of 'qidpidtriples.train.full.tsv' are shuffled in a tsv file
qidpidtriples_shuffled_file = os.path.join(write_dir, 'qidpidtriples.train.full.shuffled.tsv')



######################## Reranker data
train_qidpidlabel_file = os.path.join(reranker_dir, 'train.qidpidlabel.tsv')
valid_qidpidlabel_file = os.path.join(reranker_dir, 'validation.qidpidlabel.tsv')




######################## DPR data



######################## ColBERT data
                                     