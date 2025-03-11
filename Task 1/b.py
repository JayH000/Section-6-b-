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

# Parameters
mu = 0.1
sigma = 0.3
T = 1.0
dt = 0.01
N_paths = 5

# Simulate process
t, X = simulate_process(mu, sigma, T, dt, N_paths)

# Print the generated stochastic variable
print("Generated stochastic variable X_t:")
print(X)

# Create simulated plot

plt.figure(figsize=(10, 5))
for i in range(N_paths):
    plt.plot(t, X[i], label=f'Path {i+1}')
plt.xlabel('Time t')
plt.ylabel('X_t')
plt.title('Simulation of X_t = exp(mu*t + sigma*W_t)')
plt.legend()
plt.show()