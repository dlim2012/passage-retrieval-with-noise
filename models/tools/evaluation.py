import numpy as np

def mrr(min_ranks, k=10):
    """
    Calculate the MRR(mean reciprocal rank) given the minimum ranks and k
    """
    reciprocal_ranks_at_k = []
    for min_rank in min_ranks:
        if min_rank <= k:
            reciprocal_ranks_at_k.append(1/min_rank)
        else:
            reciprocal_ranks_at_k.append(0)
    return np.mean(reciprocal_ranks_at_k)

def recall(ranks_all, k=10):
    """
    Calculate the recall@k given all ranks and k
    """
    results = []
    for ranks in ranks_all:
        if len(ranks) == 0:
            continue
        count = 0
        for rank in ranks:
            if rank <= k:
                count += 1
        results.append(count / len(ranks))
    return np.mean(results)