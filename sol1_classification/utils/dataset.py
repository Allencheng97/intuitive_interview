from __future__ import print_function, division
import os
import pandas as pd
from skimage import transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CustomDataset(Dataset):
    def __init__(self,root_dir,csv_file,transform=None,target_transform=None) -> None:
        self.root_dir = root_dir
        self.img_labels = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.img_labels)
    
    def __getitem__(self,index) -> tuple:
        img_path =img_path = os.path.join(self.root_dir, self.img_labels.iloc[index, 1])
        image = read_image(img_path)
        label = self.img_labels.iloc[index, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

        