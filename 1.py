import gradio as gr
from PIL import Image
import numpy as np
from predict import predict

def run(img):
    if img is None:
        return {}
    
    # Gradio Sketchpad передает словарь с маской и фоном. 
    # Достаем слой 'composite', где находится сам рисунок
    if isinstance(img, dict) and 'composite' in img:
        img_data = img['composite']
        # Если это массив numpy, переводим в PIL Image
        if isinstance(img_data, np.ndarray):
            img = Image.fromarray(img_data)
        else:
            img = img_data
            
    return predict(img)

# Настройка интерфейса (Sketchpad для рисования, Label для топ-3 результатов)
demo = gr.Interface(
    fn=run,
    inputs=gr.Sketchpad(type='numpy', label="Нарисуйте цифру"),
    outputs=gr.Label(num_top_classes=3, label="Результат распознавания"),
    title='Распознавание цифр (MNIST)'
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
