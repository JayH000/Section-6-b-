import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # Set seed for reproducibility

def simulate_process(mu, sigma, T, N):
    """
    Simulate the process X_t = exp(mu * t + sigma * W_t) using different time steps.
    """
    dt = T / N  # Time step
    t = np.linspace(0, T, N+1)
    dW = np.sqrt(dt) * np.random.randn(N_paths, N)  # Wiener increments
    W = np.cumsum(dW, axis=1)  # Brownian paths
    W = np.hstack((np.zeros((N_paths, 1)), W))  # Add W_0 = 0
    
    # Compute X_t
    X = np.exp(mu * t + sigma * W)
    return t, X, dW

def integrate_ito(X, dt):
    """Compute the Itô integral of X_t using the left-point rule."""
    return np.cumsum(X[:, :-1] * dt, axis=1)

def integrate_stratonovich(X, dW, dt):
    """Compute the Stratonovich integral of X_t using the midpoint rule."""
    X_mid = 0.5 * (X[:, :-1] + X[:, 1:])  # Midpoint approximation
    return np.cumsum(X_mid * dW, axis=1)

# Parameters
mu = 0.1
sigma = 0.3
T = 10.0
N_values = np.logspace(1, 4, num=10, dtype=int)  # Log-spaced values for N
N_paths = 100

# Storage for statistics
ito_means, ito_vars = [], []
strat_means, strat_vars = [], []

for N in N_values:
    t, X, dW = simulate_process(mu, sigma, T, N)
    
    # Compute integrals
    X_ito = integrate_ito(X, T/N)
    X_stratonovich = integrate_stratonovich(X, dW, T/N)
    
    # Store mean and variance
    ito_means.append(np.mean(X_ito[:, -1]))
    ito_vars.append(np.var(X_ito[:, -1]))
    strat_means.append(np.mean(X_stratonovich[:, -1]))
    strat_vars.append(np.var(X_stratonovich[:, -1]))

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(N_values, ito_means, marker='o', label='Itô Mean')
axes[0, 0].set_xscale('log')
axes[0, 0].set_title("Itô Mean")
axes[0, 0].set_xlabel("N")
axes[0, 0].set_ylabel("Mean")

axes[0, 1].plot(N_values, ito_vars, marker='o', label='Itô Variance')
axes[0, 1].set_xscale('log')
axes[0, 1].set_title("Itô Variance")
axes[0, 1].set_xlabel("N")
axes[0, 1].set_ylabel("Variance")

axes[1, 0].plot(N_values, strat_means, marker='o', label='Stratonovich Mean')
axes[1, 0].set_xscale('log')
axes[1, 0].set_title("Stratonovich Mean")
axes[1, 0].set_xlabel("N")
axes[1, 0].set_ylabel("Mean")

axes[1, 1].plot(N_values, strat_vars, marker='o', label='Stratonovich Variance')
axes[1, 1].set_xscale('log')
axes[1, 1].set_title("Stratonovich Variance")
axes[1, 1].set_xlabel("N")
axes[1, 1].set_ylabel("Variance")

plt.tight_layout()
plt.show()
