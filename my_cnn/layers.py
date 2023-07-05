import numpy as np

from .functional import ReLU, ReLU_grad, conv_slices, pooling_slices


class Convolutional:
    def __init__(self, n_kernels, kernel_size):
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.kernel = np.random.randn(n_kernels, 
                                      kernel_size, 
                                      kernel_size) / kernel_size ** 2

    def forward(self, image):
        h, w = image.shape[:2]
        out = np.zeros((
            h - self.kernel_size + 1,
            w - self.kernel_size + 1,
            self.n_kernels
        ))
        for slice, i, j in conv_slices(image, self.kernel_size):
            out[i, j] = np.sum(slice * self.kernel,
                               axis = (1, 2))
        return ReLU(out)
    
    def backward(self, out_grad):
        activation_grad = ReLU_grad(out_grad)
        self.grad = np.zeros(self.kernel.shape)
        for slice, i, j in self.slices(self.image):
            for k in range(self.n_kernels):
                self.grad[k] += slice * activation_grad[i, j, k]
        return self.grad

    def step(self, learning_rate=1e-3):
        self.kernel -= learning_rate * self.grad


class MaxPooling:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def forward(self, image):
        h, w, n_kernels = image.shape
        out = np.zeros((
            h // self.kernel_size,
            w // self.kernel_size,
            n_kernels
        ))
        for slice, i, j in pooling_slices(image, self.kernel_size):
            out[i, j] = np.amax(slice, axis=(0, 1))
        return out
    
    def backward(self, out_grad):
        self.grad = np.zeros(self.image.shape)
        for slice, i, j in self.slices(self.image):
            h, w, n_kernels = slice.shape
            max_values = np.amax(slice, axis=(0, 1))

            for il in range(h):
                for jl in range(w):
                    for kl in range(n_kernels):
                        if slice[il, jl, kl] == max_values[kl]:
                            self.grad[
                                i * self.kernel_size + il,
                                j * self.kernel_size + jl,
                                kl
                            ] = out_grad[i, j, kl]
        return self.grad
    
    def step(self, learning_rate):
        pass


class Linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(
            input_size, output_size
        ) / np.sqrt(input_size)
        self.biases = np.zeros(output_size)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, image):
        self.original_shape = image.shape
        self.image = image.flatten()
        self.out = np.dot(self.image, self.weights) + self.biases
        return np.exp(self.out) / np.sum(np.exp(self.out), axis=0)
    
    def backward(self, out_grad):
        for i, grad in enumerate(out_grad):
            if grad:
                exp = np.exp(self.out)
                S_total = np.sum(exp)

                dy_dz = -exp[i] * exp / S_total ** 2
                dy_dz[i] = exp[i] * (S_total - exp[i]) / S_total ** 2

                dz_dw = self.image
                dz_db = 1
                dz_dinp = self.weights

                dL_dz = grad * dy_dz

                self.dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
                self.dL_db = dL_dz * dz_db
                self.grad = dz_dinp @ dL_dz

        return self.grad.reshape(self.original_shape)
    
    def step(self, learning_rate=1e-3):
        self.weights -= learning_rate * self.dL_dw
        self.biases -= learning_rate * self.dL_db
