import numpy as np

"""SIMULATION - UTILITIES"""

def damped_oscillator(T=20000, dt=0.01, zeta=0.02, omega0=2*np.pi*1.0, noise_std=0.05):
    t = np.arange(T)*dt
    x = np.exp(-zeta*omega0*t)*np.cos(omega0*np.sqrt(1-zeta**2)*t)
    x += np.random.normal(0, noise_std, size=t.shape)
    return t, x

def inject_spikes(x, n_spikes=60, amp_range=(3,6)):
    x = x.copy()
    idx = np.random.choice(len(x), size=n_spikes, replace=False)
    amps = np.random.uniform(*amp_range, size=n_spikes) * np.random.choice([-1,1], n_spikes)
    x[idx] += amps
    return x, idx

def inject_level_shifts(x, n_shifts=10, width=300, amp_range=(1.0, 2.0)):
    x = x.copy()
    starts = np.random.choice(len(x)-width-1, size=n_shifts, replace=False)
    idxs = []
    for s in starts:
        amp = np.random.uniform(*amp_range) * np.random.choice([-1,1])
        x[s:s+width] += amp
        idxs.append((s, s+width))
    return x, idxs

def inject_freq_shift(x, t, n_regions=6, width=400, scale_range=(0.6,1.6)):
    x = x.copy()
    starts = np.random.choice(len(x)-width-1, size=n_regions, replace=False)
    idxs = []
    for s in starts:
        scale = np.random.uniform(*scale_range)
        segment_t = t[:width] - t[0]
        # re-generate a segment with shifted frequency
        omega0 = 2*np.pi*scale
        zeta = 0.02
        seg = np.exp(-zeta*omega0*segment_t)*np.cos(omega0*np.sqrt(1-zeta**2)*segment_t)
        x[s:s+width] = seg + np.random.normal(0, 0.05, size=width)
        idxs.append((s, s+width))
    return x, idxs

def inject_var_burst(x, n_bursts=8, width=250, std_range=(0.5, 1.5)):
    x = x.copy()
    starts = np.random.choice(len(x)-width-1, size=n_bursts, replace=False)
    idxs = []
    for s in starts:
        std = np.random.uniform(*std_range)
        x[s:s+width] += np.random.normal(0, std, size=width)
        idxs.append((s, s+width))
    return x, idxs