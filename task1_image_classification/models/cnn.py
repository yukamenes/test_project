import tensorflow as tf 
from tensorflow import keras
import numpy as np
from models.classifier_interface import MnistClassifierInterface

class MnistCnnClassifier(MnistClassifierInterface):
    def __init__(self, input_shape=(28,28,1), num_classes=10):
        """
        Initializes the CNN model for MNIST classification.
        
        Parameters:
        - input_shape: tuple, shape of the input images (28,28,1) for grayscale MNIST images.
        - num_classes: int, number of output classes (10 for digits 0-9).
        """
        
        self.model = keras.Sequential([
            # First convolutional layer: 32 filters, 3x3 kernel, ReLU activation
            keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            # First max pooling layer: 2x2 pooling window
            keras.layers.MaxPooling2D((2, 2)), 

            # Second convolutional layer: 64 filters, 3x3 kernel, ReLU activation
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            # Second max pooling layer: 2x2 pooling window
            keras.layers.MaxPooling2D((2, 2)), 

            # Flatten the feature maps into a 1D vector for the fully connected layer
            keras.layers.Flatten(),

            # Fully connected dense layer with 128 neurons and ReLU activation
            keras.layers.Dense(128, activation="relu"),

            # Output layer with softmax activation for multi-class classification
            keras.layers.Dense(num_classes, activation="softmax")
        ])
        
        # Compile the model with Adam optimizer, sparse categorical cross-entropy loss, and accuracy metric
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    
    def train(self, x_train, y_train, epochs=5, batch_size=32):
        """
        Trains the CNN model on the given training dataset.

        Parameters:
        - x_train: np.array, training images.
        - y_train: np.array, corresponding labels.
        - epochs: int, number of training iterations over the dataset.
        - batch_size: int, number of samples per training batch.
        """
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    def predict(self, x_test):
        """
        Predicts labels for the given test dataset.

        Parameters:
        - x_test: np.array, test images.

        Returns:
        - np.array, predicted class labels.
        """
        predictions = self.model.predict(x_test)
        return np.argmax(predictions, axis=1)
