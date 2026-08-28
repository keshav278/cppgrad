import mnist
import numpy as np

mnist.datasets_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

X_train = mnist.train_images()
y_train = mnist.train_labels()
X_test = mnist.test_images()
y_test = mnist.test_labels()

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

#one-hot encoding for labels
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



print(y_train[0:5])
X_train.tofile("X_train.bin")
y_train.tofile("y_train.bin")
X_test.tofile("X_test.bin")
y_test.tofile("y_test.bin")
