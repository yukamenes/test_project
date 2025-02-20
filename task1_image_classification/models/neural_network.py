from tensorflow import keras
from models.classifier_interface import MnistClassifierInterface

class MnistNeuralNetwork(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28), num_classes=10):
        """
        Initialize the neural network model.

        :param input_shape: Tuple representing the input image dimensions (28x28 pixels).
        :param num_classes: Integer representing the number of output classes (digits 0-9).
        """

        # Build a sequential neural network model
        self.model = keras.Sequential([
            # Flatten the 28x28 input image into a 784-dimensional vector
            keras.layers.Flatten(input_shape=input_shape),
            # First fully connected (Dense) layer with 128 neurons and ReLU activation function
            keras.layers.Dense(128, activation="relu"),
            # Second fully connected layer with 64 neurons and ReLU activation function
            keras.layers.Dense(64, activation="relu"),
            # Output layer with the number of neurons equal to the number of classes (10)
            # Uses Softmax activation to convert the output into probabilities
            keras.layers.Dense(num_classes, activation="softmax")
        ])

        # Compile the model using the Adam optimizer and sparse categorical crossentropy loss function
        self.model.compile(optimizer="adam",
                           loss="sparse_categorical_crossentropy",  # Used for integer class labels
                           metrics=["accuracy"])  # Track accuracy during training

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        """
        Train the neural network model.

        :param x_train: Training images (input data).
        :param y_train: Corresponding class labels (correct answers).
        :param epochs: Number of training iterations over the entire dataset.
        :param batch_size: Number of samples processed in one batch.
        """
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, x_test):
        """
        Predict the classes for the given test images.

        :param x_test: Input images for testing.
        :return: An array of predicted class labels (digits 0-9).
        """
        predictions = self.model.predict(x_test)  # Get class probability predictions for each image
        return predictions.argmax(axis=1)  # Select the index of the most probable class for each image
