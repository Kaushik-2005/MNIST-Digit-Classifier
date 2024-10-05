import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import load_model

model, device = load_model()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def classify_digit(image):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()

interface = gr.Interface(fn=classify_digit,
                    inputs=gr.Image(type="numpy", label="Upload a handwritten digit image"),
                    outputs="label",
                    title="MNIST Digit Classifier",
                    description="Upload a 28x28 handwritten digit image, and the model will predict the digit."
)

if __name__ == "__main__":
    interface.launch()