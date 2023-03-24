import torch
import torchvision
import cv2
import os
import pandas as pd

DATA_DIR = "/hdd/zhuldyzzhan/imagenette2-320"
CLASSES = ['n02979186', 'n03417042', 'n01440764', 'n02102040', 'n03028079',
       'n03888257', 'n03394916', 'n03000684', 'n03445777', 'n03425413']

class ImagenetDataset:
    def __init__(self, items, augmentations):
        self.items = items
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        image = cv2.imread(os.path.join(DATA_DIR, item["path"]), 1)
        label = CLASSES.index(item["noisy_labels_0"])
        
        if self.augmentations:
            sample = self.augmentations(image=image)
            image = sample["image"]
        
        return image, torch.tensor(label).long()

class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="/hdd/zhuldyzzhan/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, torch.tensor(label).long()