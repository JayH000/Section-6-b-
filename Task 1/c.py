import numpy as np
import matplotlib.pyplot as plt

def simulate_process(mu, sigma, T, dt, N_paths):
    """
    
    Parameters:
        mu (float): Drift coefficient
        sigma (float): Diffusion coefficient
        T (float): Total time
        dt (float): Time step
        N_paths (int): Number of paths to simulate
    """
    N = int(T / dt)  # Number of time steps
    t = np.linspace(0, T, N+1)
    dW = np.sqrt(dt) * np.random.randn(N_paths, N)  # Wiener increments
    W = np.cumsum(dW, axis=1)  # Brownian paths
    W = np.hstack((np.zeros((N_paths, 1)), W))  # Add W_0 = 0
    
    # Compute X_t
    X = np.exp(mu * t + sigma * W)
    
    return t, X

def integrate_X(X, dt):
    """
    Compute the integral of X_t using the trapezoidal rule.
    """
    return np.cumsum(X[:, :-1] * dt, axis=1)

# Parameters
mu = 0.1
sigma = 0.3
T = 10.0
N = 100
dt = T / N
N_paths = 5

# Simulate process
t, X = simulate_process(mu, sigma, T, dt, N_paths)

# Compute integral of X_t
X_integral = integrate_X(X, dt)

# Print the generated stochastic variable
print("Generated stochastic variable X_t:")
print(X)
print("\nIntegral of X_t:")
print(X_integral)

# Plot results
plt.figure(figsize=(10, 5))
for i in range(N_paths):
    plt.plot(t[1:], X_integral[i], label=f'Path {i+1}')
plt.xlabel('Time t')
plt.ylabel('Integral of X_t')
plt.title('Integral of X_t over time')
plt.legend()
plt.show()
