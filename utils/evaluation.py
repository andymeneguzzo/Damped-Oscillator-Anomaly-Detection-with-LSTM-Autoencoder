import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""EVALUATION UTILITIES"""

@torch.no_grad()
def recon_errors(model, X):
    model.eval().to(DEVICE)
    errs = []
    crit = nn.MSELoss(reduction="none")
    for i in range(0, len(X), 1024):
        xb = torch.from_numpy(X[i:i+1024].astype(np.float32)).to(DEVICE)
        pred = model(xb)
        # per-window MSE (mean over T and feature)
        e = crit(pred, xb).mean(dim=(1,2)).detach().cpu().numpy()
        errs.append(e)
    return np.concatenate(errs, axis=0)

def evaluate_on_test(err_test, y_test, thr):
    y_pred = (err_test >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    auroc = roc_auc_score(y_test, err_test)
    ap    = average_precision_score(y_test, err_test)
    return {"precision":prec, "recall":rec, "f1":f1, "auroc":auroc, "auprc":ap}, y_pred