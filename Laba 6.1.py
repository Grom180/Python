from sympy import deg
import torch
import csv
import pandas as pd
import torch.nn as nn
import numpy as np, matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
W1, b1 = rng.normal(0, .5, (2, 16)), np.zeros((1, 16))
W2, b2 = rng.normal(0, .5, (16, 1)), np.zeros((1, 1))
sig = lambda z: 1 / (1 + np.exp(-z))
def forward(X):
    z1 = X @ W1 + b1; a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2; a2 = sig(z2)
    return z1, a1, z2, a2
def bce(p, y): return -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))
def backward(X, y, z1, a1, z2, a2):
    n = len(X)
    dz2 = a2 - y
    dW2 = a1.T @ dz2 / n; db2 = dz2.mean(0, keepdims=True)
    da1 = dz2 @ W2.T; dz1 = da1 * (1 - a1**2)
    dW1 = X.T @ dz1 / n; db1 = dz1.mean(0, keepdims=True)
    return dW1, db1, dW2, db2
# 
X_tr,X_tmp,y_tr,y_tmp=train_test_split(X,y,test_size=.3,stratify=y,random_state=42)
X_va,X_te,y_va,y_te=train_test_split(X_tmp,y_tmp,test_size=.5,stratify=y_tmp,random_state=42)
# 
w = torch.randn(2, 1, requires_grad=True)
loss = ((X_tr @ w - y_tr)**2).mean(); loss.backward(); print(w.grad)
# 
tr = DataLoader(TensorDataset(torch.tensor(X_tr.values, dtype=torch.float32), torch.tensor(y_tr.values, dtype=torch.float32)), batch_size=64, shuffle=True)
# 
class MLP(nn.Module):
    def __init__(self, d_in, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(.2), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x): return self.net(x).squeeze(1)
# 
model = MLP(X_tr.shape[1]).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()
best, wait = 1e9, 0
for ep in range(100):
    model.train()
    for xb, yb in tr:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
    model.eval(); vl = 0
    with torch.no_grad():
        for xb, yb in va: vl += loss_fn(model(xb.to(dev)), yb.to(dev)).item()
    if vl < best: best, wait = vl, 0; torch.save(model.state_dict(), 'models/mlp.pt')
    else: wait += 1
    if wait >= 5: break
# 