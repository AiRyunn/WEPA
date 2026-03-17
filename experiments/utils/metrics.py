def calc_roc_auc_score(pos_scores, neg_scores):
    from sklearn.metrics import roc_auc_score

    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = pos_scores + neg_scores
    roc_auc = roc_auc_score(y_true, y_scores)

    return max(roc_auc, 1 - roc_auc)


def calc_tpr_at_fpr(pos_scores, neg_scores, fpr_threshold=0.01):
    from sklearn.metrics import roc_curve

    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = pos_scores + neg_scores
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    tpr_at_fpr = 0.0
    for i in range(len(fpr)):
        if fpr[i] <= fpr_threshold:
            tpr_at_fpr = tpr[i]
        else:
            break

    return tpr_at_fpr
