import cupy

from .functional import ReLU, ReLU_grad, conv_slices, pooling_slices


class Convolutional:
    def __init__(self, n_kernels, kernel_size):
        self.params = {
            'n kernels': n_kernels,
            'kernel size': kernel_size
        }
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.kernel = cupy.random.randn(
            n_kernels,
            kernel_size,
            kernel_size,
            ) / kernel_size ** 2
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, image):
        self.image = image
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            _, _, self.n_channels = image.shape
        else:
            self.n_channels = 0
        out = cupy.zeros((
            h - self.kernel_size + 1,
            w - self.kernel_size + 1,
            self.n_kernels
        ))
        if self.n_channels:
            for slice, i, j in conv_slices(image, self.kernel_size):
                for channel in range(self.n_channels):
                    out[i, j] += cupy.sum(slice[:,:,channel] * self.kernel,
                                    axis=(1, 2))
        else:
            for slice, i, j in conv_slices(image, self.kernel_size):
                out[i, j] = cupy.sum(slice * self.kernel, axis=(1, 2))
        self.activation_value = ReLU(out)
        return self.activation_value
    
    def backward(self, out_grad):
        activation_grad = ReLU_grad(out_grad, self.activation_value)
        self.grad = cupy.zeros(self.kernel.shape)
        for slice, i, j in conv_slices(self.image, self.kernel_size):
            for k in range(self.n_kernels):
                for m in range(self.n_channels):
                    self.grad[k] += slice[:,:,m] * activation_grad[i, j, k]
        return self.grad

    def step(self, learning_rate=1e-3):
        self.kernel -= learning_rate * self.grad


class MaxPooling:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.params = {
            'kernel size': kernel_size,
        }

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, image):
        self.image = image
        h, w, n_kernels = image.shape
        out = cupy.zeros((
            h // self.kernel_size,
            w // self.kernel_size,
            n_kernels
        ))
        for slice, i, j in pooling_slices(image, self.kernel_size):
            out[i, j] = cupy.amax(slice, axis=(0, 1))
        return out
    
    def backward(self, out_grad):
        self.grad = cupy.zeros(self.image.shape)
        for slice, i, j in pooling_slices(self.image, 
                                          self.kernel_size):
            h, w, n_kernels = slice.shape
            max_values = cupy.amax(slice, axis=(0, 1))

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
    def __init__(self, icupyut_size, output_size):
        self.params = {
            'icupyut size': icupyut_size,
            'output size': output_size,
        }
        self.weights = cupy.random.randn(
            icupyut_size, output_size
        ) / cupy.sqrt(icupyut_size)
        self.biases = cupy.zeros(output_size)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, image):
        self.original_shape = image.shape
        self.image = image.flatten()
        self.out = cupy.dot(self.image, self.weights) + self.biases
        return cupy.exp(self.out) / cupy.sum(cupy.exp(self.out), axis=0)
    
    def backward(self, out_grad):
        for i, grad in enumerate(out_grad):
            if grad:
                exp = cupy.exp(self.out)
                S_total = cupy.sum(exp)

                dy_dz = -exp[i] * exp / S_total ** 2
                dy_dz[i] = exp[i] * (S_total - exp[i]) / S_total ** 2

                dz_dw = self.image
                dz_db = 1
                dz_dicupy = self.weights

                dL_dz = grad * dy_dz

                self.dL_dw = dz_dw[cupy.newaxis].T @ dL_dz[cupy.newaxis]
                self.dL_db = dL_dz * dz_db
                self.grad = dz_dicupy @ dL_dz

        return self.grad.reshape(self.original_shape)
    
    def step(self, learning_rate=1e-3):
        self.weights -= learning_rate * self.dL_dw
        self.biases -= learning_rate * self.dL_db
