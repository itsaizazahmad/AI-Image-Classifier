# AI Image Classifier

A simple AI project to classify images using TensorFlow and a pre-trained MobileNetV2 model.

## Purpose
This classifier identifies objects in images, trained on CIFAR-10 dataset (10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

## Technologies Used
- Python
- TensorFlow/Keras
- OpenCV
- NumPy

## How It Works
1. Train the model on CIFAR-10 using `classifier.py`.
2. Use `predict.py` to classify new images.
- The model uses a CNN architecture for feature extraction and classification.

## Installation
1. Clone this repo: `git clone https://github.com/yourusername/AI-Image-Classifier.git`
2. Install dependencies: `pip install -r requirements.txt`

## Usage
1. Train the model: `python classifier.py`
2. Classify an image: Update `image_path` in `predict.py` and run `python predict.py`

## Example Output
Predicted: cat with 85.32% confidence

## Contributing
Feel free to fork and improve! Add more classes or use custom datasets.

## License
MIT License