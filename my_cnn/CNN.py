import functools
import pickle
import warnings
import numpy as np
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score, precision_score

from .layers import Convolutional, MaxPooling, Linear


class Classifier:
    '''CNN classifier with convolutional, pooling and linear layers'''
    LAYERS = {
        'conv': Convolutional,
        'pool': MaxPooling,
        'lin': Linear,
    }
    def __init__(self, architecture, layer_params, n_classes, input_size):
        '''
        Arguments:
            architecture (str or list[str]): sequence of layer names.
            Accepted layer names: conv, pool, lin. Expected input:
                if str: layer names with spaces in between
                if list: layer names (str) in list
            
            params (list[tuple[int]]): list of parameters per layer. Expected inputs
            per layer type:
                conv (tuple): (number of kernels, kernel size)
                pool (tuple): (kernel size)
                lin  (tuple): (output size)

            n_classes (int): number of classes in data
            input_size (tuple[int] or like): size of input
        '''
        self.trained = False
        self.n_classes = n_classes
        if isinstance(architecture, str):
            architecture = architecture.split()
        assert len(architecture) == len(layer_params), 'Architecture and parameters length do not match'
        self.layers = []
        for i, layer_type in enumerate(architecture):
            if layer_type == 'lin':
                #This gets input size for a linear layer by doing a 
                # forward pass with an empty matrix
                assert self.layers, 'Linear layer can not be first'
                dummy = functools.reduce(lambda x, y: y(x), self.layers, np.zeros(input_size))
                layer_params[i] = (np.product(dummy.shape), layer_params[i])                    
            self.layers.append(self.LAYERS[layer_type](*layer_params[i]))

    def forward(self, image):
        '''Forward pass'''
        layers = [layer.forward for layer in self.layers]
        return functools.reduce(lambda x, y: y(x), layers, image)

    def backward(self, output_grad):
        '''Backpropagation'''
        layers = [layer.backward for layer in self.layers[::-1]]
        return functools.reduce(lambda x, y: y(x), layers, output_grad)

    def fit(self, X_train, y_train, learning_rate=1e-3, n_epochs=20):
        '''Training routine'''
        train_size = len(X_train)

        self.metrics = defaultdict(list)

        for epoch in range(1, n_epochs + 1):
            print(f'Epoch {epoch}/{n_epochs}')
            y_pred = []
            running_loss = []

            for sample in range(train_size):
                mean_loss = np.mean(running_loss) if running_loss else 0
                print(f'Sample {sample}/{train_size}, loss {mean_loss:.5f}', end='\r')
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

            self.metrics['accuracy'].append(accuracy_score(y_train, y_pred))
            self.metrics['balanced accuracy'].append(balanced_accuracy_score(y_train, y_pred))
            self.metrics['recall'].append(recall_score(y_train, y_pred, average='micro'))
            self.metrics['precision'].append(precision_score(y_train, y_pred, average='micro'))
            self.metrics['loss'].append(np.mean(running_loss))
            print(f'Epoch {epoch}\tloss: {np.mean(running_loss):.3f}\t',\
                    f'balanced accuracy on train: {balanced_accuracy_score(y_train, y_pred):.3f}')
            
        self.trained = True


    def step(self, learning_rate=1e-3):
        '''Learning rate application'''
        for layer in self.layers:
            layer.step(learning_rate)

    def predict(self, images):
        assert self.trained, 'The network was never trained'
        y_pred = [self.forward(image).argmax() for image in images]
        return y_pred

    def show_parameters(self):
        '''Prints a description of the network'''
        for i, layer in enumerate(self.layers, start=1):
            print(f'Layer {i}: {layer.__class__.__name__}:')
            for param_name, param_value in layer.params.items():
                print(f'\t{param_name}: {param_value}')

    def save(self, path):
        '''Saves the model as pickle'''
        if not self.trained:
            warnings.warn('Saving an untrained model')
        if not path.endswith('.pkl'):
            path += '.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self, file)
