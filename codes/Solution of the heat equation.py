import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye

D = 1  # Diffusion constant
Lx = 1  # Domain length

nsteps = 10000  # number of time steps
nout = 500  # plot every nout time steps
Nx = 500  # number of intervals
nx = Nx + 1  # number of gridpoints in x direction including boundaries
dx = 2 * Lx / Nx  # grid size in x
x = np.linspace(-Lx, Lx, nx)  # x values on the grid
dt = (dx)**2 / (2 * D)  # borderline stability of FTCS scheme

alpha = dt * D / dx**2
diagonals = [2 * (1 + alpha) * np.ones(nx), -alpha * np.ones(nx - 1), -alpha * np.ones(nx - 1)]
A = diags(diagonals, [0, -1, 1], (nx, nx)).toarray()
I = eye(nx).toarray()
A[[0, -1], :] = I[[0, -1], :]  # boundaries

sigma = Lx / 16
u = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x / sigma)**2)
u = u.reshape(-1, 1)
plt.plot(x, u)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$u(x, t)$', fontsize=14)
plt.title('Solution of the heat equation', fontsize=16)

for m in range(1, nsteps + 1):
    b = np.concatenate(([0], alpha * u[:-2] + 2 * (1 - alpha) * u[1:-1] + alpha * u[2:], [0]))
    u = np.linalg.solve(A, b)
    if m % nout == 0:
        plt.plot(x, u)

plt.show()