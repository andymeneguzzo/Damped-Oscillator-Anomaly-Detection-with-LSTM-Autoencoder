import torch
import torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, in_dim=1, hidden=64, latent=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.encoder = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
                               batch_first=True, dropout=0.0 if num_layers==1 else dropout)
        self.to_latent = nn.Linear(hidden, latent)

        self.decoder_init = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
                               batch_first=True, dropout=0.0 if num_layers==1 else dropout)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x):  # x: (B, T, 1)
        enc_out, (h, c) = self.encoder(x)  # enc_out: (B,T,H)
        # use last hidden state (from top layer)
        h_last = enc_out[:, -1, :]  # (B,H)
        z = self.to_latent(h_last)  # (B,L)

        # initialize decoder hidden from z
        h0 = torch.tanh(self.decoder_init(z)).unsqueeze(0).repeat(self.decoder.num_layers,1,1)
        c0 = torch.zeros_like(h0)

        # teacher forcing on input zeros (or the original x shifted)
        dec_in = torch.zeros_like(x)  # (B,T,1)
        dec_out, _ = self.decoder(dec_in, (h0, c0))  # (B,T,H)
        y = self.out(dec_out)  # (B,T,1)
        return y