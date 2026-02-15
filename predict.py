import tensorflow as tf
import cv2
import numpy as np

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
model = tf.keras.models.load_model('model/image_classifier.h5')

def classify_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))  # Resize to match model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return class_names[predicted_class], confidence

# Example usage: Replace 'path/to/your/image.jpg' with an actual image
if __name__ == "__main__":
    image_path = 'path/to/your/image.jpg'  # e.g., download a sample from CIFAR-10 or use your own
    label, conf = classify_image(image_path)
    print(f"Predicted: {label} with {conf*100:.2f}% confidence")