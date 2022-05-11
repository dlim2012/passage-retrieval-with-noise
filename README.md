## Preprocess files

### 1) Download data from the official website
Official website: https://microsoft.github.io/msmarco/

#### Used files

* queries.train.tsv
* queries.dev.tsv
* collection.tsv
* qrels.train.tsv
* qrels.dev.tsv
* qidpidtriples.train.full.tsv
* top1000.dev
* (Optional) triples.train.small.tsv
* (Optional) top1000.train.txt

#### Save directory


2) Save the downloaded files in the following folder: "data/ms_marco/original/"

(The base directory for text files, checkpoints, and precomputed vectors can be changed by changing the 'data_dir', 'ckpt_dir', 'vectors_dir' variables that are in the top of 'preprocess/tools/paths.py'.)


### 2) Make the main train and validation set

```bash
python3 preprocess/split_train_validation.py
```

### 3) Prepare data for evaluation

```bash
python3 preprocess/top1000.py --data_type dev
```

### 4) Generate noisy texts

```bash
python3 preprocess/noisy.py
```

(Backtranslation took several days)
```bash
# Split passage and query files into small files
python3 preprocess/noisy_backtranslation.py --make_initial_copy

# Perform backtranslation (some files may not be converted at once due to disconnection. -> Repeat)
python3 preprocess/noisy_backtranslation.py
```

## Train models

```bash
# Train the Reranker model
python3 train.py --model_name reranker --version_name v1

# Train the Dense Passage Retriever model
python3 train.py --model_name dpr --version_name v1

# Train ColBERT model
python3 train.py --model_name colbert --version_name v1
```
#### Arguments
```text
* Required arguments
--model_name: name of the model to train. ('reranker', 'dpr', 'colbert')
--version_name: name of this training that will be used to name log files and checkpoint files

* Arguments that are not required
--train_data_file: path to train data file (default: preprocessed train file)
--validation_data_file: path to validation data file (default: preprocessed train file)
--batch_size: batch size that will be used in train (default: 22)
--n_train_steps: maximum train steps (default: 200000)
--n_validation_steps: number of validation steps at each interval (default: 1000)
--lr: learning rate (default: 3e-7)
--not_lr_schedule: using this argument will disable linear learning rate scheduling
--warm_up_steps: number of warm-up steps for linear learning rate schedule (default: 20000)
--linear_decay_steps: number of steps that the learning rate becomes 0 (default: 200000)
--measure_steps: number of steps of the interval that validation is performed (default: 1000)
--n_checkpoints: number of checkpoints to save (default=0)

# Only for reranker
# (collate_fn2 truncates queries at size 64)
--collate_fn2: use this argument to change input types (default: truncate only at length 512)
--npr: negative to positive ratio for training data (npr >= 1)

# Only for ColBERT
--N_q: query token length for ColBERT (default: 32)
--vector_size: vector size of each token for ColBERT (default: 128)
```

## See training logs
```bash
# ex. To see the train log of the reranker model in localhost:1111
cd log
tensorboard --log_dir='reranker' --port 1111
```

## Evaluate reranking

### Task

Rerank using top 1000 retrieved passages using BM25. Some queries may not have exactly 1000 passages retrieved using BM25.

Evaluation: MRR@10, Recall@10

### Code example
```bash
python evaluate_rerank.py --model_name reranker --ckpt_version v1 --ckpt_name steps=116000_loss-interval=0.2303.pth
```
### Arguments
```text
--model_name: (Required) the name of the model ('reranker', 'dpr', 'colbert')

--ckpt_dir: directory to save checkpoints (default: ckpt_dir saved in preprocess/tools/paths)
--ckpt_version: the name of the train version
--ckpt_name: the name of the checkpoint
--mode: one of the followings.
    'original': measure base performance
    'noisy_queries': measure performance on noisy queries
    'noisy_passages': measure performance on noisy passages
--batch_size: batch size that will be used in train (default: 22)

# Only for reranker
# (collate_fn2 truncates queries at size 64)
--collate_fn2: use this argument to change input types (default: truncate only at length 512)

# Only for ColBERT
--N_q: query token length for ColBERT (default: 32)
--vector_size: vector size of each token for ColBERT (default: 128)
```

## Precompute

precompute dense vectors for the 8.8 million passages in the 'collection.tsv' file using a trained dpr model.

### Code example
```bash
python precompute.py --ckpt_version v1 --ckpt_name steps=116000_loss-interval=0.2303.pth
```

### Arguments
--model_name: (Not required) the name of the model (only available option: 'dpr')

--ckpt_dir: directory to save checkpoints (default: ckpt_dir saved in preprocess/tools/paths)
--ckpt_version: the name of the train version 
--ckpt_name: the name of the checkpoint
--vectors_dir: the directory that all vector files are saved. (default: vectors_dir saved in preprocess/tools/paths)
    (ckpt_name will be used to locate the file assuming that the file was saved through precompute.py)
    
--batch_size: batch size that will be used in train (default: 22)
--mode: one of the followings.
    'original': measure base performance
    'noisy_queries': measure performance on noisy queries
    'noisy_passages': measure performance on noisy passages
```
## Evaluate retrieval

Task: retrieve passages from a collection of about 8.8 million passages

Evaluation: MRR@10, Recall@1000
### Code example
```bash
python evaluate_retrieval.py --ckpt_version v1 --ckpt_name steps=116000_loss-interval=0.2303.pth
```
### Arguments
```text
--model_name: (Not required) the name of the model (only available option: 'dpr')

--ckpt_dir: directory to save checkpoints (default: ckpt_dir saved in preprocess/tools/paths)
--ckpt_version: the name of the train version 
--ckpt_name: the name of the checkpoint
--vectors_dir: the directory that all vector files are saved. (default: vectors_dir saved in preprocess/tools/paths)
    (ckpt_name will be used to locate the file assuming that the file was saved through precompute.py)
    
--batch_size: batch size that will be used in train (default: 22)
--mode: one of the followings.
    'original': measure base performance
    'noisy_queries': measure performance on noisy queries
    'noisy_passages': measure performance on noisy passages
```
## Others

### Clean Latin-1 and some other characters
```bash
python3 preprocess/clean_latin.py # result: latin_cleaned_dir saved in preprocess/tools/paths
```

### Make dataset with about top 20~30 BM25 results as "hard negatives"
(Using this dataset didn't give better results)
```bash
python3 preprocess/top1000_bm25.py # Rerank top 1000, needed about 60-64GB of memory and a few days
python3 preprocess/top1000_bm25_dataset.py # Make the training dataset
```
