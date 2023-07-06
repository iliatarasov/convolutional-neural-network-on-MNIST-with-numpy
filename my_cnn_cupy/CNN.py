import functools
import cupy
import PIL
from collections import defaultdict
import numpy as np

from .layers import Convolutional, MaxPooling, Linear


class Classifier:
    LAYERS = {
        'conv': Convolutional,
        'pool': MaxPooling,
        'lin': Linear,
    }
    def __init__(self, architecture, params, n_classes):
        self.trained = False
        self.n_classes = n_classes
        if isinstance(architecture, str):
            architecture = architecture.split()
        assert len(architecture) == len(params), 'Architecture and parameters length do not match'
        self.layers = []
        for i, layer_type in enumerate(architecture):
            self.layers.append(self.LAYERS[layer_type](*params[i]))

    def forward(self, image):
        layers = [layer.forward for layer in self.layers]
        return functools.reduce(lambda x, y: y(x), layers, image)

    def backward(self, output_grad):
        layers = [layer.backward for layer in self.layers[::-1]]
        return functools.reduce(lambda x, y: y(x), layers, output_grad)

    def fit(self, dataset, learning_rate=1e-3,
            n_epochs=20, show_progress=True):
        train_size = len(dataset)

        if show_progress == True:
            show_progress = 1

        self.metrics = defaultdict(list)

        for epoch in range(1, n_epochs + 1):
            print(f'Epoch {epoch}')
            y_pred = []
            running_loss = []

            for sample in range(train_size):
                print(sample)
                image = dataset[sample]['img']
                if isinstance(image, PIL.PngImagePlugin.PngImageFile):
                    image = cupy.asarray(image)

                label = cupy.array(dataset[sample]['label'])
                prediction = self.forward(image)

                loss = -cupy.log(prediction[label])
                loss_grad = cupy.zeros(self.n_classes)
                loss_grad[label] = -1 / prediction[label]

                self.backward(loss_grad)
                self.step(learning_rate)
                y_pred.append(prediction.argmax())

                running_loss.append(loss.get())

            self.metrics['loss'].append(np.mean(running_loss))
            

        self.trained = True


    def step(self, learning_rate=1e-3):
        for layer in self.layers:
            layer.step(learning_rate)

    def predict(self, image):
        assert self.trained, 'The network was never trained'
        prediction = self.forward(image)
        return prediction.argmax()

    def show_parameters(self):
        for i, layer in enumerate(self.layers, start=1):
            print(f'Layer {i}: {layer.__class__.__name__}:')
            for param_name, param_value in layer.params.items():
                print(f'\t{param_name}: {param_value}')


