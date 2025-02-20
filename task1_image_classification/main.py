import numpy as np
import tensorflow as tf 
from mnist_classifier import MnistClassifier

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (scale to range 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a classifier (Random Forest in this case)
classifier = MnistClassifier("nn")

# Train the model on the training dataset
classifier.train(x_train, y_train)

# Predict labels for the test dataset
predictions = classifier.predict(x_test)

# Calculate the accuracy of predictions
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")