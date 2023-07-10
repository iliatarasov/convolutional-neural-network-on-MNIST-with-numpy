import functools
import numpy as np
from collections import defaultdict

from .layers import Convolutional, MaxPooling, Linear


class Classifier:
    '''CNN classifier with convolutional, pooling and linear layers'''
    LAYERS = {
        'conv': Convolutional,
        'pool': MaxPooling,
        'lin': Linear,
    }
    def __init__(self, architecture, params, n_classes):
        '''
        Arguments:
            architecture (str or list[str]): sequence of layer names:
                if str: layer names with spaces in between
                if list: layer names (str) in list
            params (list): list of parameters per layer. Expected inputs
            for all layer types:
                conv: (number of kernels, kernel size)
                pool: (kernel size)
                lin:  (input size, output size)
        '''
        self.trained = False
        self.n_classes = n_classes
        if isinstance(architecture, str):
            architecture = architecture.split()
        assert len(architecture) == len(params), 'Architecture and parameters length do not match'
        self.layers = []
        for i, layer_type in enumerate(architecture):
            self.layers.append(self.LAYERS[layer_type](*params[i]))

    def forward(self, image):
        '''Forward pass'''
        layers = [layer.forward for layer in self.layers]
        return functools.reduce(lambda x, y: y(x), layers, image)

    def backward(self, output_grad):
        '''Backpropagation'''
        layers = [layer.backward for layer in self.layers[::-1]]
        return functools.reduce(lambda x, y: y(x), layers, output_grad)

    def fit(self, X_train, y_train, learning_rate=1e-3,
            n_epochs=20, show_progress=True):
        '''Training routine'''
        train_size = len(X_train)

        if show_progress == True:
            show_progress = 1

        self.metrics = defaultdict(list)

        for epoch in range(1, n_epochs + 1):
            print(f'Epoch {epoch}/{n_epochs}')
            y_pred = []
            running_loss = []

            for sample in range(train_size):
                mean_loss = np.mean(running_loss) if running_loss else 0
                print(f'Sample {sample}/{train_size}, loss {mean_loss}', end='\r')
                image = X_train[sample]

                label = np.array(y_train[sample])
                prediction = self.forward(image)

                loss = -np.log(prediction[label])
                loss_grad = np.zeros(self.n_classes)
                loss_grad[label] = -1 / prediction[label]

                self.backward(loss_grad)
                self.step(learning_rate)
                y_pred.append(prediction.argmax())

                running_loss.append(loss)
            print()
            self.metrics['loss'].append(np.mean(running_loss))
            

        self.trained = True


    def step(self, learning_rate=1e-3):
        '''Learning rate application'''
        for layer in self.layers:
            layer.step(learning_rate)

    def predict(self, image):
        assert self.trained, 'The network was never trained'
        prediction = self.forward(image)
        return prediction.argmax()

    def show_parameters(self):
        '''Prints a description of the network'''
        for i, layer in enumerate(self.layers, start=1):
            print(f'Layer {i}: {layer.__class__.__name__}:')
            for param_name, param_value in layer.params.items():
                print(f'\t{param_name}: {param_value}')


