from layers import Convolutional, MaxPooling, Linear


class Classifier:
    def __init__(self):
        self.layers = [
            Convolutional(),
            MaxPooling(),
            Linear()
        ]

    def fit(self):
        
