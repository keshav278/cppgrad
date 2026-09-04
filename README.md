# cppgrad

A small C++20 automatic-differentiation and neural-network experiment. The included MNIST example trains a 784-16-16-10 residual MLP using ReLU activations, softmax, cross-entropy loss, and stochastic gradient descent. Based on [MagicalBat's YT vid](https://youtu.be/hL_n_GljC0I)

## Requirements

- C++20 compiler
- CMake 3.21+
- Python 3 with `mnist` and `numpy`

## Build and run

From this directory:

```powershell
python -m pip install mnist numpy
python get_mnist_data.py
cmake -S . -B build
cmake --build build --config Release
.\build\Release\mnist.exe
```

The data script creates `X_train.bin`, `y_train.bin`, `X_test.bin`, and `y_test.bin` in the project directory. The program expects these files in its working directory, prints a sample digit, trains for three epochs, and reports test accuracy and cost.

## Layout

- `mat.h`: matrix storage and numerical operations
- `graph.h`: computation graph, forward pass, and gradients
- `mnist.cpp`: model definition, training loop, and executable entry point
- `get_mnist_data.py`: MNIST download and binary conversion


