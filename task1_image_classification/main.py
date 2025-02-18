import numpy as np
import tensorflow as tf 
from mnist_classifier import MnistClassifier


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


classifier = MnistClassifier("rf")


classifier.train(x_train, y_train)


predictions = classifier.predict(x_test)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")