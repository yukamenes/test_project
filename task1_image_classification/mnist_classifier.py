from models.classifier_interface import MnistClassifierInterface
from models.random_forest import MnistRandomForest
from models.neural_network import MnistNeuralNetwork
from models.cnn import MnistCnnClassifier

class MnistClassifier:
    """
    Unified classifier for MNIST dataset.
    
    This class allows selecting a classification algorithm (Random Forest, Neural Network, CNN)
    and provides a unified interface for training and prediction.
    """
    def __init__(self, algorithm:str):
        """
        Initialize the classifier with the specified algorithm.

        Parameters:
        - algorithm: str
          The name of the classification algorithm ('rf', 'nn', 'cnn').
        """
        if algorithm == "rf":
            self.model = MnistRandomForest()
        elif algorithm == "nn":
            self.model = MnistNeuralNetwork()
        elif algorithm == "cnn":
            self.model = MnistCnnClassifier()
        else:
            raise ValueError("Unsupported algorithm. Choose from: 'rf', 'nn', 'cnn'.")
    def train(self, x_train, y_train):
        """
        Train the selected classifier.

        Parameters:
        - X_train: array-like
          Training data (images).
        - y_train: array-like
          Training labels.
        """
        self.model.train(x_train, y_train)
    def predict(self, x_test):
        """
        Predict using the selected classifier.

        Parameters:
        - X_test: array-like
          Test data (images).

        Returns:
        - array-like
          Predicted labels.
        """
        return self.model.predict(x_test)