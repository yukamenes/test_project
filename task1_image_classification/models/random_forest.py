from sklearn.ensemble import RandomForestClassifier
from models.classifier_interface import MnistClassifierInterface


class MnistRandomForest(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST dataset.

    This class implements a Random Forest model to classify handwritten digits from MNIST.
    It follows the MnistClassifierInterface structure with train() and predict() methods.
    """
    def __init__(self, n_estimators = 100, random_state = 42):
        """
        Initialize the Random Forest classifier.

        Parameters:
        - n_estimators: int, default=100
        The number of trees in the forest.
        - random_state: int, default=42
        Random seed for reproducibility.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    def train(self, x_train, y_train):
        """
        Train the Random Forest classifier on the MNIST dataset.

        Parameters:
        - X_train: array-like, shape (n_samples, n_features)
        The training input samples (flattened images).
        - y_train: array-like, shape (n_samples,)
        The target labels corresponding to the training samples.
        """
        # Reshape the images into 1D feature vectors
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train_flat, y_train)
    def predict(self, x_test):
        """
        Predict labels for the given test dataset.

        Parameters:
        - X_test: array-like, shape (n_samples, n_features)
        The test input samples (flattened images).

        Returns:
        - y_pred: array-like, shape (n_samples,)
        The predicted labels for the test samples.
        """
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        return self.model.predict(x_test_flat)


