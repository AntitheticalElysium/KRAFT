import numpy as np

def precision_at_k(y_true_items, y_pred_items, k):
    """
    Calculates Precision@k.
    y_true_items: set of relevant item IDs for a user.
    y_pred_items: list of recommended item IDs, ordered by relevance.
    k: number of recommendations to consider.
    """
    pred_k = y_pred_items[:k]
    relevant_in_pred_k = len(set(pred_k) & y_true_items)
    return relevant_in_pred_k / k if k > 0 else 0

def recall_at_k(y_true_items, y_pred_items, k):
    """
    Calculates Recall@k.
    y_true_items: set of relevant item IDs for a user.
    y_pred_items: list of recommended item IDs, ordered by relevance.
    k: number of recommendations to consider.
    """
    pred_k = y_pred_items[:k]
    relevant_in_pred_k = len(set(pred_k) & y_true_items)
    return relevant_in_pred_k / len(y_true_items) if len(y_true_items) > 0 else 0

def ndcg_at_k(y_true_items, y_pred_items, k):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG)@k.
    y_true_items: set of relevant item IDs for a user.
    y_pred_items: list of recommended item IDs, ordered by relevance.
    k: number of recommendations to consider.
    """
    pred_k = y_pred_items[:k]
    dcg = 0.0
    for i, item in enumerate(pred_k):
        if item in y_true_items:
            # 1 if relevant, 0 otherwise)
            dcg += 1.0 / np.log2(i + 2) # +2 since index starts at 0

    # Assumes all true items are ranked at the top
    idcg = 0.0
    num_relevant = min(len(y_true_items), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0
