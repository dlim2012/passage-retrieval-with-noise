from datasets import load_dataset
from transformers import BertTokenizer
import os
import numpy as np

# Directory to save preprocessed data
data_dir = 'Data/MS_Marco/rerank/'

# Download MS Marco dataset
ms_marco = load_dataset('ms_marco', 'v2.1')

# Model name
model_name = 'bert-base-uncased'

# BERT Tokenizer from HuggingFace
tokenizer = BertTokenizer.from_pretrained(model_name) # WordPiece tokenizer


for dataset_type in ['train', 'validation']:

    ms_marco_type = ms_marco[dataset_type]
    n = len(ms_marco_type) # n_train: 808731, n_validation: 101093

    ###############################################################################
    #if dataset_type == 'train':
    #    n = 100
    #else:
    #    n = 10
    ###############################################################################

    input_ids = []
    labels = []

    for i in range(n):

        # Truncate query to have at most 64 tokens
        query_tokens = tokenizer.tokenize(ms_marco_type[i]['query'])[:64]

        # Convert tokens of query to token ids
        query_token_ids = tokenizer.convert_tokens_to_ids(query_tokens)

        query_len = len(query_token_ids)

        passages = ms_marco_type[i]['passages']['passage_text']

        for passage in passages:

            # Truncate passage so that [<CLS> query <SEP> passage <SEP>] doesn't exceed 512
            passage_tokens = tokenizer.tokenize(passage)[:509 - query_len]

            # Convert tokens of passage to token ids
            passage_token_ids = tokenizer.convert_tokens_to_ids(passage_tokens)

            # token ids for input
            token_ids = [101] + query_token_ids + [102] + passage_token_ids + [102]

            # pad token ids to use batch
            token_ids += [0] * (512 - len(token_ids))

            input_ids.append(token_ids)

        labels += ms_marco_type[i]['passages']['is_selected']


    input_ids_save_path = os.path.join(data_dir, '%s_input_ids.npy' % dataset_type)
    labels_save_path = os.path.join(data_dir, '%s_labels.npy' % dataset_type)

    np.save(input_ids_save_path, input_ids)
    np.save(labels_save_path, labels)


