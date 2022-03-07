from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch
import glob, os
import random

class AirfoilDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, train:bool=True,reduce_set:float=1):
        super(AirfoilDataset, self).__init__(root, transform, pre_transform)
        self.train=train
        self.root=root
        self.data, self.slices = torch.load(self.processed_paths[0])
        if (self.train):
            self.filenames = glob.glob(os.path.join(self.root,'train_*.pt'))
            self.filenames = random.sample(self.filenames, int(len(self.filenames)*reduce_set))
        else:
            self.filenames = glob.glob(os.path.join(self.root,'test_*.pt'))
            self.filenames = random.sample(self.filenames, int(len(self.filenames)*reduce_set))
        self.transform = transform
        for filename in self.filenames:
            if (transform):                
                self.data += [self.transform(x) for x in torch.load(filename)]
            else:
                self.data += torch.load(filename)

    @property
    def raw_file_names(self):
        if (self.filenames):
            return self.filenames
        else:
            return []        

    @property
    def processed_file_names(self):
        if (self.filenames):
            return self.filenames
        else:
            return [] 

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        pass