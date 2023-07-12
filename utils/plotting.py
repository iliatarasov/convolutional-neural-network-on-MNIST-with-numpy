import numpy as np
import matplotlib.pyplot as plt


def plot_multiple(images: np.ndarray, shared_title: str=None, 
                  figsize: tuple[int]=None) -> None:
    '''
    Utility function that plots multiple images contained in
    a single numpy array with dimensions [:, :, n_images]
    Arguments:
        images (np.ndarray): array of images
        shared_titme (str): shared title for all images
        figsize (tuple[int]): figsize for the plot
    '''
    n_images = images.shape[2]
    if figsize is None:
        figsize = (10, 5)
    _, ax = plt.subplots(1, n_images, figsize=figsize)
    for i in range(n_images):
        ax[i].imshow(images[:,:,i], cmap='gray')
        ax[i].axis('off')
        if shared_title is not None:
            ax[i].set_title(f'{shared_title} {i+1}')
