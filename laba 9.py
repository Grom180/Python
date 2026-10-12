from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons

import numpy as np, matplotlib.pyplot as plt
import torch
import torch.nn as nn

# -
loss_fn = nn.BCEWithLogitsLoss()
Xm,ym=make_moons(300,noise=.2,random_state=0)
Xt=torch.tensor(Xm,dtype=torch.float32); yt=torch.tensor(ym,dtype=torch.float32)
# -

rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 1, 30)); y = np.sin(2 * np.pi * X) + rng.normal(0, .3, 30)
Xv = np.sort(rng.uniform(0, 1, 30)); yv = np.sin(2 * np.pi * Xv) + rng.normal(0, .3, 30)
for d in [1, 4, 15]:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(X[:, None], y)
    print(d, np.mean((m.predict(X[:, None]) - y)**2), np.mean((m.predict(Xv[:, None]) - yv)**2))
# -
def fit(model, opt, steps=2000):
    tr, va = [], []
    # -
    Xva=torch.tensor(Xm,dtype=torch.float32); yva=torch.tensor(ym,dtype=torch.float32)
    # -
    for i in range(steps):
        model.train(); opt.zero_grad(); l = loss_fn(model(Xt).squeeze(1), yt); l.backward(); opt.step()
        if i % 50 == 0:
            model.eval()
            with torch.no_grad(): tr.append(l.item()); va.append(loss_fn(model(Xva).squeeze(1), yva).item())
    return tr, va
big = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
tr, va = fit(big, torch.optim.Adam(big.parameters(), 1e-2))
plt.plot(tr, label='train'); plt.plot(va, label='val'); plt.legend(); plt.show()
# -
reg = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128,128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128,1))
tr2, va2 = fit(reg, torch.optim.Adam(reg.parameters(), 1e-2, weight_decay=1e-2))
