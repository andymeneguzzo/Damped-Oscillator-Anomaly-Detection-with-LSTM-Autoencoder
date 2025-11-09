from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, roc_curve
import numpy as np
from evaluation import recon_errors, evaluate_on_test

"""THRESHOLDING"""


def threshold_percentile_on_train(model, X_train, perc=99.5):
    errs_train = recon_errors(model, X_train)
    return np.percentile(errs_train, perc)

def threshold_youden(err_val, y_val):
    fpr, tpr, thr = roc_curve(y_val, err_val)
    J = tpr - fpr
    j_idx = np.argmax(J)
    return thr[j_idx]

def tune_threshold(err_val, y_val, strategy="f1"):
    # search across percentiles of errors
    percentiles = np.linspace(50, 99.9, 400)
    best = {"thr": None, "f1": -1, "prec":0, "rec":0}
    for p in percentiles:
        thr = np.percentile(err_val, p)
        y_pred = (err_val >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best = {"thr": thr, "f1": f1, "prec":prec, "rec":rec}
    # also compute AUROC and AUPRC (threshold-free)
    auroc = roc_auc_score(y_val, err_val)
    ap    = average_precision_score(y_val, err_val)
    return best, auroc, ap


def eval_with(thr, err_test, y_test):
    m, _ = evaluate_on_test(err_test, y_test, thr)
    return m