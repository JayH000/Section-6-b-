import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 0.5  # Damping coefficient
D = 1.0      # Diffusion coefficient
T = 10.0     # Total time
N = 1000     # Number of steps
dt = T / N   # Time step
v0 = 1.0     # Initial velocity
M = 1000     # Number of realizations

# Time array
t = np.linspace(0, T, N+1)

# Initialize velocity array
v = np.zeros((M, N+1))
v[:, 0] = v0

# Wiener process (Brownian motion)
dW = np.sqrt(dt) * np.random.randn(M, N)

# Simulate the Langevin equation using Euler-Maruyama
for i in range(N):
    eta = (2 * np.cumsum(dW[:, :i+1], axis=1)[:, -1] + dt * (i+1)) / dt  # Approximation of η(t)
    v[:, i+1] = v[:, i] + (-gamma * v[:, i] + np.sqrt(2 * D) * eta) * dt

# Compute statistics
mean_v = np.mean(v, axis=0)
var_v = np.var(v, axis=0)

# Plot results
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

axs[0].plot(t, mean_v, label='Mean velocity')
axs[0].set_ylabel("Mean v(t)")
axs[0].legend()

axs[1].plot(t, var_v, label='Variance of velocity', color='r')
axs[1].set_xlabel("Time t")
axs[1].set_ylabel("Var v(t)")
axs[1].legend()

plt.suptitle("Mean and Variance of Velocity from Langevin Equation")
plt.show()

