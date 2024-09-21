import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os 

class Dataset(Dataset):
    def __init__(self, label_file, transform=None):
        """
        Args:
            label_file (string): 标签文件的路径。
            transform (callable, optional): 应用于图像的可选转换。
        """
        self.transform = transform
        self.image_pairs = []
        with open(label_file, 'r') as file:
            for line in file:
                path1, path2, label = line.strip().split()
                self.image_pairs.append((path1, path2, int(label)))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_path1, img_path2, label = self.image_pairs[idx]
        image1 = Image.open(img_path1).convert('RGB')
        image2 = Image.open(img_path2).convert('RGB')
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return image1, image2, torch.tensor(label, dtype=torch.float32)



