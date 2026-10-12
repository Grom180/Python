import torch, torch.nn as nn, matplotlib.pyplot as plt, time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.set_num_threads(4)  # число ядер процессора

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train = datasets.MNIST('data', train=True, download=True, transform=tf)
test = datasets.MNIST('data', train=False, download=True, transform=tf)
img, label = train[0]; print(img.shape, label)
fig, ax = plt.subplots(5, 5, figsize=(6, 6))
for i, a in enumerate(ax.flat): a.imshow(train[i][0][0], cmap='gray'); a.set_title(train[i][1]); a.axis('off')
plt.show()
# показывает картинку где нейронка угадывает какое число написано
train_dl = DataLoader(train, batch_size=128, shuffle=True)
test_dl = DataLoader(test, batch_size=512)
x, y = next(iter(train_dl)); print(x.shape, y.shape, len(train_dl))
# -
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
# -
for ep in range(3):
    t0 = time.time(); model.train()
    for x, y in train_dl:
        opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_dl: correct += (model(x).argmax(1) == y).sum().item()
    sched.step()
    print(f'epoch {ep} loss {loss.item():.3f} acc {correct / len(test):.4f} lr {sched.get_last_lr()[0]} time {time.time()-t0:.0f}s')
# и скачивает на пк картикнки и архивы этих картинок