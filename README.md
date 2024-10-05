# MNIST Digit Classifier with CNN in PyTorch

This project is an implementation of a Convolutional Neural Network (CNN) to classify handwrittern digits from the MNIST dataset. The project includes both the trainig process and an interactive ui where users can upload their own handwritten digits and get a classification result.

## Features:
- **CNN Model:** A Convolutional Neural Network built using PyTorch for accurate digit classification.
- **Training:** Includes a script to train the model on the MNIST dataset and save the model for further use.
- **Gradio Interface:** A user-friendly UI built using Gradio, allowing users to upload an image of a handwritten digit and classify it using the trained CNN model.
- **CUDA Support:** The trainig and inference scripts support GPU acceleration if available.

## Requirements:
- PyTorch
- Gradio
- Torchvision
- Pillow

## Structure:
```bash
.
├── app.py            # Contains the Gradio-based UI for digit classification
├── model.py          # Contains the CNN model architecture and model loading logic
├── train.py          # Script to train the CNN model on the MNIST dataset
├── README.md         # Project documentation
├── mnist_cnn.pth     # (Generated after training) Trained model file (run train.py)
└── data/             # MNIST dataset downloaded automatically (run train.py)
```

## Installation:
1. Clone the repository:
```bash
git clone https://github.com/Kaushik-2005/MNIST-Digit-Classifier.git
cd mnist-digit-classifier
```
2. Install the required dependencies:
```bash
pip install torch torchvision gradio pillow
```

## Training the Model:
To train the CNN on the MNIST dataset, run the following command:
```bash
python train.py
```
This will train the model and save it as `mnist_cnn.pth` in the currect directory. The training script automatically downloads the MNIST dataset if it is not already avaliable in `./data`.

## Running the App:
Once the model is trained, you can use the Gradio app to upload and classify handwritten digits.<br>
Run the following command:
```bash
python app.py
```
This will start a local web server with a Gradio interface. Open your browser and go to the URL provided in the terminal (usually `http://127.0.0.1:7860/`). You can then upload a 28*28 images of a handwritten digit, and the model will predict the digit.

## Model Architecture:
The CNN used in the project consists of the following layers:
- Two convolutional layers followed by ReLU activation and max pooling:<br>
&emsp; - Conv1: 1 input channel(grayscale), 32 output channels, kernel size 3*3. <br>
&emsp; - Conv2: 32 input channels, 64 output channels, kernel size 3*3.
- A fully connected (dense) layer:<br>
&emsp; - FC1: 64*7*7 inputs, 128 outputs. <br>
&emsp; - FC2: 128 inputs, 10 outputs(for digits 0-9).<br>

## CUDA Support:
The code automatically detects if a CUDA-enabled GPU is available and uses it for training and inference. If not, it falls back to the CPU.