import numpy as np
import matplotlib.pyplot as plt

# Script to solve for an ellipse model with 5 sample points [x,y]
# https://www.youtube.com/watch?v=H7EwHBN9Vus citation for positive definite

# Five given points (x, y)
pts = np.array([
    [2.92, -6.01],
    [3.40, -7.20],
    [4.99, -7.84],
    [5.48, -7.04],
    [4.20, -5.91],
])

x = pts[:, 0]
y = pts[:, 1]

# 5x5 matrix of system
M = np.column_stack([x**2, 2*x*y, y**2, -2*x, -2*y])

# Right hand constants matrix
r = -1*np.ones(5)

# Solve for coefficients
A, B, C, D, E = np.linalg.solve(M, r)
print("A: ", A, " B: ",B," C: ", C," D: ",D," E: ", E)

# Solve for b and Q
P = np.array([[A, B], [B, C]])
d = np.array([D, E])
Pinv = np.linalg.inv(P)
 
alpha = 1.0 / (d @ Pinv @ d - 1.0)
Q = alpha * P
b = Pinv @ d 

eigvals = np.linalg.eigvalsh(Q)
print("eigenvalues of Q:", eigvals, "-> ellipse" if np.all(eigvals > 0) else "-> NOT an ellipse")
print(Q)
print(b)
print(alpha)

# Points on the ellipse: p(t) = b + Q^{-1/2} [cos t, sin t]^T
lam, V = np.linalg.eigh(Q)
Q_inv_sqrt = V @ np.diag(1.0 / np.sqrt(lam)) @ V.T
t = np.linspace(0, 2*np.pi, 400)
circle = np.vstack([np.cos(t), np.sin(t)])          # 2 x 400
ellipse = b.reshape(2, 1) + Q_inv_sqrt @ circle     # 2 x 400

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(ellipse[0], ellipse[1], "b-", label="Fitted ellipse")
ax.scatter(x, y, c="r", zorder=3, label="Data points")
ax.plot(*b, "k+", markersize=12, label="Center b")
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Ellipse fit through 5 points")
ax.grid(True)
ax.legend()
plt.savefig("q1b_ellipse_plot.png", dpi=150)
plt.show()

