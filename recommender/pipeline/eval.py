"""offline AUC metric"""

from sklearn.metrics import roc_auc_score


def offline_auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))
