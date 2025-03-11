import numpy as np

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

def integrate_ito(X, dW):
    """Compute the Itô integral using the left-point rule."""
    return np.cumsum(X[:, :-1] * dW, axis=1)

def integrate_stratonovich(X, dW):
    """Compute the Stratonovich integral using the midpoint rule."""
    X_mid = 0.5 * (X[:, :-1] + X[:, 1:])  # Midpoint approximation
    return np.cumsum(X_mid * dW, axis=1)

# Parameters
mu = 0.1
sigma = 0.3
T = 10.0
N = 100
N_paths = 5

t, X, dW = simulate_process(mu, sigma, T, N)

# Compute Itô and Stratonovich integrals
I_ito = integrate_ito(X, dW)
I_strat = integrate_stratonovich(X, dW)

# Compute correction term (1/2 * sigma * Itô integral of dX)
dX = np.diff(X, axis=1)
correction_term = 0.5 * sigma * np.cumsum(dX, axis=1)

# Verify conversion: Stratonovich = Itô + correction term
I_strat_approx = I_ito + correction_term

# Print results for first trajectory
print("Itô Integral (last value):", I_ito[0, -1])
print("Stratonovich Integral (last value):", I_strat[0, -1])
print("Itô + Correction Term (last value):", I_strat_approx[0, -1])