from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    Interface for MNIST classification models.

    This interface defines a common structure for all classifiers 
    used to train and predict on the MNIST dataset.
    
    Each classifier must implement:
    - `train(x_train, y_train)`: Trains the model on the given dataset.
    - `predict(x_test)`: Predicts labels for the given test dataset.
    """

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Train the classifier on the provided training dataset.

        Parameters:
        - x_train: array-like, shape (n_samples, n_features)
          The training input samples (images).
        - y_train: array-like, shape (n_samples,)
          The target labels corresponding to the training samples.
        """
        pass

    @abstractmethod
    def predict(self, x_test):
        """
        Predict labels for the given test dataset.

        Parameters:
        - x_test: array-like, shape (n_samples, n_features)
          The test input samples (images).

        Returns:
        - y_pred: array-like, shape (n_samples,)
          The predicted labels for the test samples.
        """
        pass



