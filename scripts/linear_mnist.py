from models.linear import Linear

from file_handling.load_datasets import load_mnist

(X_train, Y_train), (X_test, Y_test) = load_mnist()
linear_model = Linear(X_train.shape[1], Y_train.shape[1])

train_accuracy = linear_model.train_model(X_train, Y_train, .1)
print(f"Train Accuracy: {train_accuracy}")

test_accuracy = linear_model.test_model(X_test, Y_test)
print(f"Test Accuracy: {test_accuracy}")