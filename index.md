---
layout: default
title: LSTM Autoencoder for Anomaly Detection on Damped Oscillator Dataset - MSE loss vs Physics-Informed loss
---
{% include mathjax.html %}


# LSTM Autoencoder for Anomaly Detection on Damped Oscillator Dataset - MSE loss vs Physics-Informed loss

## Abstract
Anomalies are data points that deviate, subtly or evidently, from normal behaviour. They can occur in a variety of contexts: economics, cybersecurity, manifacturing, robotics and many others. Detecting anomalies is a fundamental task to prevent potential financial consequences, information loss and damage to systems or people. In this project, I study **reconstruction-based Anomaly Detection** on simulated **damped harmonic oscillator** data training an **LSTM Autoencoder**, comparing traditional **mean squared error (MSE) loss** to a **Physics-informed loss** that embeds the system's dynamics equation. Normal data is simulated from a noisy underdamped harmonic oscillator. Anomalies (spikes, level shifts, frequency shifts and variance bursts) are inserted and labelled at window level after overlapping segmentation (128-size window, stride 32) and standardization. The baseline LSTM Autoencoder is first trained with MSE loss. Afterwards it is trained with a total loss that sums MSE and an ODE residual embedding the damped oscillator equation, encouraging reconstructions with physical knowledge. The two models are evaluated via **F1-maximizing reconstruction-error thresholding** on validation data. On test data, both models present strong performance with **97.3% AUROC and 97.7% AUPRC**. **Physics-informed loss slightly improves F1 score, compared to MSE loss, from 89% to 89.4%**, demonstrating better sensitivity to anomalies thanks to it's dynamics knowledge. 

## Introduction
When studying the behaviour of **dynamical systems**, it's fundamental to be able to robustly analyze **time series data** and understand temporal dependencies and constraints imposed by dynamical equations. With the right tools it is possible to detect anomalies in time series data and potential prevent failures, flag sensor faults and regime shifts. In this project, thanks to it's capability of understanding time-sequantial data, I use LSTM Autoencoder to learn the **manifold** of normal dynamics and flag as anomalous temporal windows with high reconstruction error. I then compare coventional training with **MSE loss** to a **Physics informed loss** that augments MSE with the **damped oscillator differential equation**. 

An **LSTM (Long Short-Term Memory)** is a **recurrent neural network (RNN)** architecture designed to model sequential data with long-range dependencies. Information flow through time is controlled by it's **gates: Input, Forget and Output**, allowing the model to conserve or remove information and context as needed. It's architecture is suited especially suited for:
- forecasting and filtering time-sequential signals
- learning dynamics in noisy settings measured in timesteps
- compressing high dimentional data (encoding) into compact lower-dimensional latent representation and understand normal behaviour

In a **reconstruction-based anomaly detection** setup, an LSTM Autoencoder is trained to reconstruct normal behaviour. When model is tested, windows that deviate from learned dynamics give higher **reconstruction errors** and are flagged as anomalies if the errors exceed the set threshold. The training methods studied are:
- **Standard MSE loss**: model **minimizes the pointwise reconstruction error between the input and it's reconstruction**. Though being simple, it's generally stable and widely applicable. However, it presents limitations in understanding the dynamics of the system, so every reconstruction minimizing MSE is considered acceptable, even if not physically normal.
- **Physics-informed loss**: add MSE loss term to a residual penalty that embeds the damped oscillator equation $x'' + 2\zeta\omega_0 x' + \omega_0^2 x = 0$. To achieve this, I approximate $x'$ and $x''$ with **finite differences** on the reconstructed sequence and penalize the squared residual, thus obtaining the total loss:

$$
L = \mathrm{MSE}(x, \hat{x}) + \lambda_{\mathrm{phys}} \cdot \left\| x'' + 2\zeta\omega_0 x' + \omega_0^2 x \right\|^2
$$

where $\lambda_{\mathrm{phys}}$ controls how much the physics loss contributes to total loss estimation, thus enforcing the autoencoder to understand reconstruction that are both close to the input but also physically consistent with system's dynamics, **improving sensitivity** to physically implausable deviations.  

Comparing these objectives on the same dataset, I can assess wether pure MSE loss or embedded physical knowledge into the training can improve anomaly detection without lossing performance.

## Dataset
I simulate the **dynamics of a 1-dimensional noisy damped harmonic oscillator**. Dynamics of the system are governed by the equation:

$$
x(t) = e^{-\zeta \omega_0 t} \cos\!\left(\omega_0 \sqrt{1 - \zeta^2}\, t\right) + \varepsilon(t), 
\qquad \varepsilon(t) \sim \mathcal{N}(0, \sigma^2)
$$

where $\zeta$ is the damping ratio, $\omega_0$ the natural angular frequency and $\varepsilon(t)$ the Gaussian noise. I generate $T = 60,000$ samples spanning $t = 0$ to $t = 599.99$ with $dt = 0.01$ step. I then inject four types of anomalies: 
- **Spikes**: isolated points with abnormal impulses $x_i + a \rightarrow x_i$, with $a \in [2.5, 5.5]$
- **Level shifts**: 400-width segments are shifted by constant offset
- **Frequency shifts**: 500-width segments are resimulated with scaled frequency $\omega_0 \cdot \kappa \rightarrow \omega_0, \kappa \in [0.5, 1.7]$
- **Variance bursts**: 300-width segments gain elevated noise $\varepsilon(t)$

The feature analyzed is the displacement $x_t$. To preprocess the data I create **windowed samples** with:
- length $W = 128$ and stride $s = 32$
- for each start index $k$, I extract $X_k = (x_k, x_{k+1},...,x_{k+W-1}) \in \mathbb{R}^{128 \times 1}$
- window label $y_k$ is set to 1 if at least 5% of it's points are anomalous

$$
y_k = \mathbf{1}\!\left( \frac{1}{W} \sum_{j=0}^{W-1} y_{k+j}^{\text{point}} \ge 0.05 \right)
$$

otherwise $y_k = 0$. After windowing process, I get 1,872 windows with approximately anomaly rate ≈ 31.7%.

I split normal windows in:
- 60% training normals
- 20% test normals
- 20% validation normals

I split anomaly windows in:
- 50% test anomalies
- 50% validation anomalies

Finally, before model training, (normal) data is **scaled** using ```StandardScaler``` on the flattened windows and **normalized** so that training distribution has zero mean and unit variance

$$
\tilde{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
$$

The plot below shows the displacement $x(t)$ with respect to $t$ (already containing anomalies). 

<p align="center">
  <img src="plots/Damped_Oscillator_Data.png" width="90%">
</p>



## LSTM Autoencoder model


## MSE loss vs. Physics-Informed loss

## Implementation

## Results
