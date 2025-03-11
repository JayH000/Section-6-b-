import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # Seed from b

def simulate_process(mu, sigma, T, dt, N_paths):
    N = int(T / dt)  # Number of time steps
    t = np.linspace(0, T, N+1)
    dW = np.sqrt(dt) * np.random.randn(N_paths, N)  # Wiener increments
    W = np.cumsum(dW, axis=1)  # Brownian paths
    W = np.hstack((np.zeros((N_paths, 1)), W))  # Add W_0 = 0
    
    # Compute X_t
    X = np.exp(mu * t + sigma * W)
    
    return t, X, dW
def integrate_stratonovich(X, dW, dt):
    X_mid = 0.5 * (X[:, :-1] + X[:, 1:])  # Midpoint approximation
    return np.cumsum(X_mid * dW, axis=1)

# Parameters
mu = 0.1
sigma = 0.3
T = 10.0
N = 100
dt = T / N
N_paths = 5

# Simulate process
t, X, dW = simulate_process(mu, sigma, T, dt, N_paths)

# Compute Stratonovich integral of X_t
X_stratonovich = integrate_stratonovich(X, dW, dt)

# Print results
print("Generated stochastic variable X_t:")
print(X)
print("\nStratonovich integral of X_t:")
print(X_stratonovich)

# Plot results
plt.figure(figsize=(10, 5))
for i in range(N_paths):
    plt.plot(t[1:], X_stratonovich[i], label=f'Path {i+1}')
plt.xlabel('Time t')
plt.ylabel('Stratonovich Integral of X_t')
plt.title('Stratonovich Integral of X_t over time')
plt.legend()
plt.show()
