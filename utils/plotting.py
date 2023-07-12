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
    if figsize is None:
        figsize = (10, 5)
    n_images = images.shape[2]
    if len(images.shape) == 2:
        plt.imshow(images, cmap='gray', figsize=figsize)
        plt.title(shared_title)
        return
    _, ax = plt.subplots(1, n_images, figsize=figsize)
    for i in range(n_images):
        ax[i].imshow(images[:,:,i], cmap='gray')
        ax[i].axis('off')
        if shared_title is not None:
            ax[i].set_title(f'{shared_title} {i+1}')

def plot_metrics(model: object) -> None:
    '''
    Utility function that plots metrics of a neural network
    of a class Classifier of my_cnn module
    '''
    n_metrics = len(model.metrics)
    fig, ax = plt.subplots(n_metrics, 1, figsize=(10, 3.5*n_metrics))
    fig.tight_layout(pad=5)
    for i, metric in enumerate(model.metrics):
        n_epochs = len(model.metrics[metric])
        ax[i].plot(range(1, n_epochs + 1), model.metrics[metric], color='k')
        ax[i].axhline(model.metrics[metric][-1], linestyle='--', color='g')
        ax[i].set_title(metric)
        ax[i].set_xlabel('epoch')