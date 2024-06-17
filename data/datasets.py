import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Datasets(Dataset):
    def __init__(self, dataset_path):
        self.data_dir = dataset_path
        self.imgs = []
        self.imgs += glob(os.path.join(self.data_dir, '*.jpg'))
        self.imgs += glob(os.path.join(self.data_dir, '*.png'))
        self.imgs.sort()
        if len(self.imgs) == 0:
            raise ValueError(f"Dataset path {self.data_dir} is empty")
        self.transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, item):
        image_path = self.imgs[item]
        name = os.path.basename(image_path)
        image = Image.open(image_path).convert('RGB')
        img = self.transform(image)
        return img, name

    def __len__(self):
        return len(self.imgs)


def get_test_loader(test_dir, batch_size=1, shuffle=False):
    test_dataset = Datasets(test_dir)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle)
    return test_loader
