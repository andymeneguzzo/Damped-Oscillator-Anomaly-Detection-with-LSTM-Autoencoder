import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""TRAINING UTILITIES"""

# Train with MSE loss criterion
def train_autoencoder(model, train_loader, val_loader, epochs=20, lr=1e-3, weight_decay=1e-5, clip=1.0):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss(reduction="mean")

    hist = {"train_loss":[], "val_loss":[]}
    best_state, best_val = None, float("inf")

    for ep in range(1, epochs+1):
        model.train()
        tr_losses = []
        for xb in train_loader:
            xb = xb.to(DEVICE)
            pred = model(xb)
            loss = crit(pred, xb)
            opt.zero_grad()
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tr_losses.append(loss.item())
        train_loss = float(np.mean(tr_losses))

        # val on validation normals only (approximate by filtering val normal windows)
        model.eval()
        with torch.no_grad():
            v_losses = []
            for xb in val_loader:
                xb = xb.to(DEVICE)
                pred = model(xb)
                v_losses.append(crit(pred, xb).item())
            val_loss = float(np.mean(v_losses))

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

        print(f"Epoch {ep:03d} | train {train_loss:.5f} | val {val_loss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


# Train with Physics Informed Loss (damped oscillator equation)
def train_autoencoder_phys(model, train_loader, val_loader,
                           epochs=20, lr=1e-3, weight_decay=1e-5, clip=1.0,
                           dt=0.01, zeta=0.02, omega0=2*np.pi*1.0, lambda_phys=0.1):
    """
    Physics-informed training for LSTM Autoencoder of a damped oscillator system.
    Adds a physics loss term enforcing: x'' + 2*zeta*omega0*x' + omega0^2*x = 0
    """

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss(reduction="mean")

    hist = {"train_loss": [], "val_loss": [], "phys_loss": []}
    best_state, best_val = None, float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses, phys_losses = [], []

        for xb in train_loader:
            xb = xb.to(DEVICE)
            pred = model(xb)

            # Reconstruction loss 
            loss_recon = crit(pred, xb)

            # Physics-informed loss
            # Approximate derivatives using central differences
            x = pred  # shape [batch, seq_len]
            dx = (x[:, 2:] - x[:, :-2]) / (2 * dt)
            ddx = (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]) / (dt ** 2)

            # Trim x to align shapes with derivatives
            x_mid = x[:, 1:-1]

            residual = ddx + 2 * zeta * omega0 * dx + (omega0 ** 2) * x_mid
            loss_phys = torch.mean(residual ** 2)

            # Total loss
            loss = loss_recon + lambda_phys * loss_phys

            opt.zero_grad()
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            tr_losses.append(loss_recon.item())
            phys_losses.append(loss_phys.item())

        train_loss = float(np.mean(tr_losses))
        train_phys = float(np.mean(phys_losses))

        # Validation
        model.eval()
        with torch.no_grad():
            v_losses = []
            for xb in val_loader:
                xb = xb.to(DEVICE)
                pred = model(xb)
                v_losses.append(crit(pred, xb).item())
            val_loss = float(np.mean(v_losses))

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["phys_loss"].append(train_phys)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {ep:03d} | train {train_loss:.5f} | phys {train_phys:.5f} | val {val_loss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, hist


def plot_learning_curves(hists, titles):
    plt.figure(figsize=(12,4))
    for h, title in zip(hists, titles):
        plt.plot(h["train_loss"], label=f"{title} Train", alpha=0.8)
        plt.plot(h["val_loss"], label=f"{title} Val", alpha=0.8, linestyle="--")
    plt.title("Learning Curves"); plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.legend(); plt.show()