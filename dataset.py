import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import pandas as pd
import numpy as np
from PIL import Image
import os
import random

disease_categories = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Effusion': 2,
    'Infiltration': 3,
    'Mass': 4,
    'Nodule': 5,
    'Pneumonia': 6,
    'Pneumothorax': 7,
    'Consolidation': 8,
    'Edema': 9,
    'Emphysema': 10,
    'Fibrosis': 11,
    'Pleural_Thickening': 12,
    'Hernia': 13,
}

class CXR8_train(Dataset):
    def __init__(self, root_dir, transform = None):
        self.image_dir = os.path.join(root_dir, 'images')
        self.index_dir = os.path.join(root_dir, 'train_label.csv')
        self.classes = pd.read_csv(self.index_dir, header=None,nrows=1).iloc[0, :].to_numpy()[1:9]
        self.label_index = pd.read_csv(self.index_dir, header=0)
        #self.bbox_index = pd.read_csv(os.path.join(root_dir, 'BBox_List_2017.csv'), header=0)
        self.transform = transform
        
    def __len__(self):
        return int(len(self.label_index)*0.1)
    
    def __getitem__(self, idx):
        name = self.label_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = Image.open(img_dir).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        label = self.label_index.iloc[idx, 1:9].to_numpy().astype('int')
        return image, label

    
class CXR8_validation(Dataset):
    def __init__(self, root_dir, transform = None):
        self.image_dir = os.path.join(root_dir, 'images')
        self.index_dir = os.path.join(root_dir, 'val_label.csv')
        self.classes = pd.read_csv(self.index_dir, header=None,nrows=1).iloc[0, :].to_numpy()[1:9]
        self.label_index = pd.read_csv(self.index_dir, header=0)
        #self.bbox_index = pd.read_csv(os.path.join(root_dir, 'BBox_List_2017.csv'), header=0)
        self.transform = transform
        
    def __len__(self):
        return int(len(self.label_index)*0.1)
    
    def __getitem__(self, idx):
        name = self.label_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = Image.open(img_dir).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        label = self.label_index.iloc[idx, 1:9].to_numpy().astype('int')
        return image, label
