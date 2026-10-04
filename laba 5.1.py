from sklearn.datasets import make_moons

import numpy as np, matplotlib.pyplot as plt

X, y = make_moons(500, noise=.2, random_state=0); y = y.reshape(-1, 1)
# plt.scatter(X, y); plt.show()
# -
rng = np.random.default_rng(0)
W1, b1 = rng.normal(0, .5, (2, 16)), np.zeros((1, 16))
W2, b2 = rng.normal(0, .5, (16, 1)), np.zeros((1, 1))
sig = lambda z: 1 / (1 + np.exp(-z))
def forward(X):
    z1 = X @ W1 + b1; a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2; a2 = sig(z2)
    return z1, a1, z2, a2
def bce(p, y): return -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))
# -
def backward(X, y, z1, a1, z2, a2):
    n = len(X)
    dz2 = a2 - y
    dW2 = a1.T @ dz2 / n; db2 = dz2.mean(0, keepdims=True)
    da1 = dz2 @ W2.T; dz1 = da1 * (1 - a1**2)
    dW1 = X.T @ dz1 / n; db1 = dz1.mean(0, keepdims=True)
    return dW1, db1, dW2, db2
# -
hist = []
for ep in range(2000):
    z1, a1, z2, a2 = forward(X)
    g = backward (X, y, z1, a1, z2, a2)
    W1 -= .5*g[0]; b1 -= .5*g[1]; W2 -= .5*g[2]; b2 -= .5*g[3]
    hist.append(bce(a2, y))  
# -
print('------')