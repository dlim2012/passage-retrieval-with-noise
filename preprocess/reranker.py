"""
train:
    Shuffle qidpidtriples.train.full.tsv > qidpidtriples.train.full.shuffled.tsv
    Read each line in qidpidtriples.train.full.shuffled.tsv
        add positive pair with probability 0.25
        add negative pair always
        save 'pid qid label' > 'train.qidpidlabel.tsv'
    convert 'train.qidpidlabel.tsv' to npy
        save 800000 in each file -- total 16
        input ids(length 512) > 'train_input_ids_npr4_{}.npy'
        labels > 'train_labels_npr4_{}.npy'

    
validation: all pairs in top1000
    'qid pid label' > 'validation.qidpidlabel.tsv'
    convert 'validation.qidpidlabel.tsv' to npy
        save 800000 in each file -- total 16
        input ids(length 512) > 'validation_input_ids_npr4_{}.npy'
        labels > 'validation_labels_npr4_{}.npy'
"""

"""
Another possible approach for train: (from typos-aware-passage-retrieval)
    For each positive qid-pid pair generate 4 negative qid-pid pair
    To make the positive pair be trained more evenly(?)
"""

#from datasets import load_dataset
from transformers import BertTokenizer
import os
import numpy as np
import csv
import random
import sys

from tools.paths import *
from tools.tools import shuffle_text, one_hot_vector
from tools.read_files import read_texts, read_qidpidtriples, read_top1000_npy, read_qrels

input_ids_file_name = 'reranker/{dataset_type}_input_ids_npr{neg_per_pos}_{save_number}.npy'
labels_file_name = 'reranker/{dataset_type}_labels_npr{neg_per_pos}_{save_number}.npy'

# Model name of the pre-trained BERT
model_name = 'bert-base-uncased'

# negative - positive ratio
neg_per_pos = 4

# Save every {} data instance (due to memory issue)
save_size = 800000
max_size = 12800000

#save_size = 100
#max_size = 1000

def format_input_ids(query, passage, tokenizer):
    """
    Given a query and a passage, use tokenizer to format reranker inputs
    Format: "<CLS> query <SEP> passage <SEP>"
        where query <= 64 tokens and total <= 512 tokens
    """

    # Truncate query to have at most 64 tokens
    query_tokens = tokenizer.tokenize(query)[:64]

    # Convert tokens of query to token ids
    query_token_ids = tokenizer.convert_tokens_to_ids(query_tokens)
    
    # Truncate passage so that [<CLS> query <SEP> passage <SEP>] doesn't exceed 512
    passage_tokens = tokenizer.tokenize(passage)[:509 - len(query_token_ids)]

    # Convert tokens of passage to token ids
    passage_token_ids = tokenizer.convert_tokens_to_ids(passage_tokens)

    # Token ids for input
    token_ids = [101] + query_token_ids + [102] + passage_token_ids + [102]

    # Pad token ids to use batch
    token_ids += [0] * (512 - len(token_ids))
    
    return token_ids

def save_in_npy(qidpidlabel_file, queries, passages, tokenizer, dataset_type):
    """
    Given a 'qid pid label' file, make and save data for reranker
    input_ids file: reranker/{dataset_type}_input_ids_npr{neg_per_pos}_{save_number}.npy
    labels file: reranker/{dataset_type}_labels_npr{neg_per_pos}_{save_number}.npy
    
    """
    
    
    print("\tsave_size: %d, max_size: %d" % (save_size, max_size))

    read_file = open(qidpidlabel_file, 'r')
    reader = csv.reader(read_file, delimiter='\t')
    
    # Lists to save data
    input_ids, labels = [], []
    
    # Save number for each file
    save_number = 0
                             
    for i, line in enumerate(reader):
                             
        qid, pid, label = map(int, line)
        
        
        input_ids.append(format_input_ids(queries[qid], passages[pid], tokenizer))
        labels.append(label)
                                     
        
        # Save every save_step
        if (i+1) % save_size == 0:
            
            # Path to save npy files
            input_ids_save_path = os.path.join(write_dir, input_ids_file_name.format(
                dataset_type=dataset_type,
                neg_per_pos=neg_per_pos,
                save_number=save_number))
            labels_save_path = os.path.join(write_dir, labels_file_name.format(
                dataset_type=dataset_type,
                neg_per_pos=neg_per_pos,
                save_number=save_number))
           
            # Check overflow
            assert np.max(input_ids) <= 32767

            # Save the results
            np.save(input_ids_save_path, np.array(input_ids, dtype=np.int16)) # np.int16: -32_768 to 32_767
            np.save(labels_save_path, one_hot_vector(labels).astype(np.int8)) # np.int8: -128 to 127
            
            print("\t%s, %s saved." % (input_ids_save_path, labels_save_path))
            
            save_number += 1
            input_ids, labels = [], []

        
        # Stop at max_step
        if (i+1) == max_size:
            break
            
    read_file.close()
    
    


def preprocess_train_from_qidpidtriples(tokenizer, passages):
    # Make write directory if not exists
    os.makedirs(write_dir, exist_ok=True)
    
    # Shuffle 'qidpidtriples.train.full.txt' if not shuffled
    if not os.path.exists(qidpidtriples_shuffled_file):
        print("\tShuffling %s and saving to %s..." % (qidpidtriples_file, qidpidtriples_shuffled_file))
        shuffle_text(qidpidtriples_file, train_qidpidlabel_file)
    else:
        print("\t%s exists." % qidpidtriples_shuffled_file)
        
    if not os.path.exists(train_qidpidlabel_file):
        print("\tWriting to %s..." % train_qidpidlabel_file)
        
        # Open shuffled qidpidtriples
        read_file = open(qidpidtriples_shuffled_file, 'r')
        reader = csv.reader(read_file, delimiter='\t')
        
        #
        write_file = open(train_qidpidlabel_file, 'w')
        
        for i, line in enumerate(reader):

            # Get ids in each line
            qid, pid1, pid2 = map(int, line)

            # Add a positive pair with pre-defined probability
            if random.random() < 1/neg_per_pos:
                write_file.write('%d\t%d\t%d\n' % (qid, pid1, 1))

            # Add a negative pair every step
            write_file.write('%d\t%d\t%d\n' % (qid, pid2, 0))
            
        read_file.close()
        write_file.close()
        
        print("\t\tShuffling lines...")
        shuffle_text(train_qidpidlabel_file, train_qidpidlabel_file)
    else:
        print("\t%s exists." % train_qidpidlabel_file)
    
    # Read queries and passages 
    queries = read_texts(train_query_file)
    
    save_in_npy(train_qidpidlabel_file, queries, passages, tokenizer, 'train')
    

    
def preprocess_validation(tokenizer, passages, top1000, positives, qidpidlabel_file, dataset_type):
    """
    From top1000 make all possible qidpidlabel triples for validation dataset
    positives: (dict)[qid] = [relevant pids]
    qidpidlabel_file: 'qid\tpid\tlabel' format
    dataset_type: used forfile name 'train', 'validation'
    """
    if not os.path.exists(qidpidlabel_file):
        print("\tWriting to %s..." % qidpidlabel_file)
    
        lines = []

        for qid in top1000:

            pids = top1000[qid]

            for pid in pids:
                if pid == -1:
                    break

                if pid in positives[qid]:
                    label = 1
                else:
                    label = 0

                lines.append('%d\t%d\t%d\n' %(qid, pid, label))

        print("\t\tShuffling lines...")
        random.shuffle(lines)

        with open(qidpidlabel_file, 'w') as f:
            f.writelines(lines)
    else:
        print("\t%s exists." % qidpidlabel_file)
    
    
    queries = read_texts(dev_query_file)
    
    save_in_npy(valid_qidpidlabel_file, queries, passages, tokenizer, dataset_type)
    
def main():
    
    os.makedirs(reranker_dir, exist_ok=True)
    
    # BERT Tokenizer from HuggingFace
    tokenizer = BertTokenizer.from_pretrained(model_name) # WordPiece tokenizer
    
    passages = read_texts(passage_file)
    #passages = dict()
    
    
    # Preprocess reranking (label: irrelevant(0), relevant(1))
    print("Preprocess reranking training data... (label: relevance)")
    print("\tneg_per_pos: %d" % neg_per_pos)
    preprocess_train_from_qidpidtriples(tokenizer, passages)
    
    # Preprocess reranking (label: irrelevant(0), relevant(1))
    print("Preprocess reranking validation data... (label: relevance)")
    top1000 = read_top1000_npy(top1000_dev_id_file)
    positives = read_qrels(dev_qrels_file)
    preprocess_validation(tokenizer, passages, top1000, positives, valid_qidpidlabel_file, 'validation')
    
    
    # TODO: preprocess reranking with top10 as label
        # TODO: save top 10 passages using HuggingFace ms_marco dataset
    

if __name__ == '__main__':
    main()
    
