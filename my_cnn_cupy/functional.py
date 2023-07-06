import cupy


def ReLU(x):
    '''Relu activation function'''
    return cupy.maximum(0, x)

def ReLU_grad(output_grad, output):
    '''Relu activation function gradient'''
    return output_grad * (output > 0).astype(float)

def conv_slices(image, kernel_size):
    '''A generator that yields image slices for convolution'''
    height, width = image.shape[:2]
    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            slice = image[
                i:(i + kernel_size),
                j:(j + kernel_size)]
            yield slice, i, j    

def pooling_slices(image, kernel_size):
    '''A generator that yields image slices for pooling'''
    h = image.shape[0] // kernel_size
    w = image.shape[1] // kernel_size
    for i in range(h):
        for j in range(w):
            slice = image[
                i * kernel_size : (i + 1) * kernel_size,
                j * kernel_size : (j + 1) * kernel_size
            ]
            yield slice, i, j


            