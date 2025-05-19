import numpy as np
from Layer_Dense import Layer_Dense
from CategoricalCrossentropyLoss import CategoricalCrossentropyLoss
from activations.ReLU import ReLU
from activations.Softmax import Softmax
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#simple INPUTS example
X = [   # shape (3, 4)
        [1.0, 2.0, 3.0, 2.5],     
        [2.0, 5.0, -1.0, 2.0],   
        [-1.5, 2.7, 3.3, -0.8]
]

X, y = spiral_data(samples=100, classes=3) #Initialize nnfs data from the spiral example
# X shape -> (300, 2)  100 per class each class 2 features

#create a Layer_Dense
dense1 = Layer_Dense(n_inputs=2, n_neurons=5)  # notice the outputs have to match the inputs in layer 2
activation1 = ReLU()

dense2 = Layer_Dense(n_inputs=5, n_neurons=3)  # inputs come from layer 1
activation2 = Softmax()

#forward passes
dense1.forward(inputs=X)
activation1.forward(dense1.output)

dense2.forward(inputs=dense1.output)
activation2.forward(dense2.output) # get first 5 rows from predicted values

print(activation2.output)

loss_function = CategoricalCrossentropyLoss()

data_loss = loss_function.calculate(output=activation2.output, y=y)


print(data_loss)