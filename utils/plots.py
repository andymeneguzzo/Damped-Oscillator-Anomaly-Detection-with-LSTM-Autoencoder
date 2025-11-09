import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""PLOTS"""

def plot_roc_pr_curves(errors_list, y_true, labels, title_suffix="(Test)"):
    plt.figure(figsize=(12,4))
    # ROC
    plt.subplot(1,2,1)
    for err, lbl in zip(errors_list, labels):
        fpr, tpr, _ = roc_curve(y_true, err)
        plt.plot(fpr, tpr, label=f"{lbl} (AUROC={roc_auc_score(y_true, err):.3f})")
    plt.plot([0,1],[0,1],'--', lw=1)
    plt.title(f"ROC Curves {title_suffix}"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()

    # PR
    plt.subplot(1,2,2)
    for err, lbl in zip(errors_list, labels):
        prec, rec, _ = precision_recall_curve(y_true, err)
        plt.plot(rec, prec, label=f"{lbl} (AUPRC={average_precision_score(y_true, err):.3f})")
    plt.title(f"Precision-Recall Curves {title_suffix}"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
    plt.show()


def plot_reconstruction_examples(model, X, n=3, title="Reconstruction examples"):
    model.eval().to(DEVICE)
    idxs = np.random.choice(len(X), size=n, replace=False)
    with torch.no_grad():
        xb = torch.from_numpy(X[idxs].astype(np.float32)).to(DEVICE)
        yb = model(xb).cpu().numpy()
    plt.figure(figsize=(12, 3*n))
    for i, idx in enumerate(idxs, 1):
        plt.subplot(n,1,i)
        plt.plot(X[idx,:,0], label="input", lw=1)
        plt.plot(yb[i-1,:,0], label="recon", lw=1)
        plt.title(f"{title}: window #{idx}")
        plt.legend()
    plt.tight_layout(); plt.show()

