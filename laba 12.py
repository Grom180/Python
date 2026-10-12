from torchvision import datasets, transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader

import torch, torch.nn as nn

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
tr = datasets.ImageFolder(r'hymenoptera_data\train', tf); va = datasets.ImageFolder(r'hymenoptera_data\val', tf)
print(tr.classes, len(tr), len(va))
# -
body = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT); body.classifier = nn.Identity(); body.eval()
def extract(ds):
    xs, ys = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, 32): xs.append(body(x)); ys.append(y)
    return torch.cat(xs), torch.cat(ys)
Ftr, ytr = extract(tr); Fva, yva = extract(va)
print(Ftr.shape)
torch.save((Ftr, ytr, Fva, yva), 'features.pt')
# -
head = nn.Linear(576, 2); opt = torch.optim.Adam(head.parameters(), 1e-3); loss_fn = nn.CrossEntropyLoss()
for ep in range(100):
    opt.zero_grad(); loss_fn(head(Ftr), ytr).backward(); opt.step()
print('val acc:', (head(Fva).argmax(1) == yva).float().mean().item())
print(sum(p.numel() for p in head.parameters()), 'обучаемых параметров')
# -
body.classifier = head; body.eval()
img, label = va[0]
print(body(img[None]).argmax(1).item(), label)
torch.save(body.state_dict(), 'ants_bees.pt')
# -сосчитал кол-во мурашек и пчил ,выдал длины классов, аккуратночть выдал, выдал кол-во обучаемых параметров,вывел тело с фото
