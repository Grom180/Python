import torch, torch.nn as nn, numpy as np
import gradio as gr

from PIL import Image, ImageOps
from PIL import Image
from predict import predict




model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
model.load_state_dict(torch.load('mnist.pt')); model.eval()
def predict(img):
    img = ImageOps.grayscale(img).resize((28, 28))
    x = np.array(img, dtype=np.float32) / 255
    if x.mean() > 0.5: x = 1 - x
    x = (x - 0.1307) / 0.3081
    with torch.no_grad(): p = model(torch.tensor(x)[None, None]).softmax(1)[0]
    return {str(i): float(p[i]) for i in range(10)}
# -
def run(img):
    if isinstance(img, dict): img = Image.fromarray(img['composite'])
    return predict(img)
gr.Interface(run, gr.Sketchpad(type='numpy'), gr.Label(num_top_classes=3), title='Распознавание цифр').launch()
