import cupy


class DataLoader:
    '''Parent class for loading data to GPU'''
    def __get__(self, idx):
        '''Returns requested item and label by idx'''
        return self.images[idx], self.labels[idx]
    
    def __len__(self):
        '''Returns length of the dataset'''
        return len(self.images)
class FromDataset(DataLoader):
    '''Loads data from a datasets library dataset'''
    def __init__(self, dataset, keys={
        'image': 'img',
        'label': 'label',
    }):
        '''
        Arguments:
            dataset (datasets.arrow_dataset.Dataset): dataset
            keys (dict): keys for accessing images and labels
            '''
        self.images = cupy.array(
            [cupy.asarray(entry[keys['image']]) for entry in dataset]
        )
        self.labels = cupy.array(
            [cupy.asarray(entry[keys['label']]) for entry in dataset]
        ) 
        assert len(self.images) == len(self.labels), 'Images and labels mismatch'
        print(f'Data loaded to device: {self.images.device}') 
    
class FromNumpy(DataLoader):
    '''Loads data from numpy arrays'''
    def __init__(self, images, labels):
        '''
        Arguments:
            images (numpy.ndarray or like): images
            labels (numpy.ndarray or like): labels
        '''
        self.images = cupy.array(images)
        self.labels = cupy.array(labels)
        assert len(self.images) == len(self.labels), 'Images and labels mismatch'
        print(f'Data loaded to device: {self.images.device}') 

