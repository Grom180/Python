from transformers import pipeline

import os
import torch

clf = pipeline('sentiment-analysis', model='cointegrated/rubert-tiny-sentiment-balanced')
for t in ['Отличный курс!', 'Ничего не понял', 'Нормально, но долго']:
    print(t, clf(t))
# -
tr = pipeline('translation', model='Helsinki-NLP/opus-mt-ru-en')
print(tr('Нейросеть — это цепочка матричных умножений.')[0]['translation_text'])
# -
from huggingface_hub import constants
print(constants.HF_HUB_CACHE)
for d in os.listdir(constants.HF_HUB_CACHE): print(d)
# -
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
w = MobileNet_V3_Small_Weights.DEFAULT; m = mobilenet_v3_small(weights=w).eval(); tf = w.transforms()
img = Image.open('photo.jpg').convert('RGB')
with torch.no_grad(): p = m(tf(img).unsqueeze(0)).softmax(1)[0]
for v, i in zip(*p.topk(5)): print(f'{w.meta["categories"][i]:25s} {v:.3f}')
# - конвертирует один тип фото в другой