import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps

# Инициализация архитектуры
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
# Загрузка весов
model.load_state_dict(torch.load('mnist.pt', map_location=torch.device('cpu')))
model.eval()

def predict(img):
    # Приведение к оттенкам серого и размеру 28x28
    img = ImageOps.grayscale(img).resize((28, 28))
    x = np.array(img, dtype=np.float32) / 255.0
    
    # Инверсия, если фон светлый
    if x.mean() > 0.5:
        x = 1.0 - x
        
    # Стандартное MNIST-масштабирование
    x = (x - 0.1307) / 0.3081
    
    # Конвертация в тензор и инференс
    tensor_input = torch.tensor(x, dtype=torch.float32)[None, None]
    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = logits.softmax(dim=1)[0]
        
    # ВОТ ЭТА СТРОКА ДОЛЖНА БЫТЬ ОБЯЗАТЕЛЬНО:
    return {str(i): float(probabilities[i]) for i in range(10)}