import numpy as np


def ReLU(x):
    '''Relu activation function'''
    return np.maximum(0, x)

def ReLU_grad(output_grad, output):
    '''Relu activation function gradient'''
    return output_grad * (output > 0).astype(float)

def convolve_2d(image, kernel):
    height, width = image.shape
    n_kernels, _, kernel_size = kernel.shape

    out = np.zeros((
        height - kernel_size + 1,
        width - kernel_size + 1,
        n_kernels
    ))
    for slice, i, j in slices(image, kernel_size):
        out[i, j] = np.sum(slice * kernel,
                           axis = (1, 2))

def slices(image, kernel_size):
    height, width = image.shape
    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            image_slice = image[
                i:(i + kernel_size),
                j:(j + kernel_size)]
            yield image_slice, i, j    


            