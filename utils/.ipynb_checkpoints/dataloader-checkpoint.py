import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image

class CaptchaDataset(Dataset):
    def __init__(self, path, transform=transforms.ToTensor()):
        self.transform = transform
        self.x, self.y = self.loadlist(path)
    
    def loadlist(self, path):
        x = []
        y = []
        for filename in glob.glob(path+'/*'):
            x.append(filename)
            y.append(filename.split('/')[2].split('.')[0])
        return x, y
    
    def str_to_id(self, y):
        keys = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        dic = {k:i for i, k in enumerate(keys)}
        ids = list(map(lambda x: dic[x], y))
        return ids
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        img = Image.open(x)
        image = self.transform(img)
        img.close()
        
        y = self.y[idx]
        ids = self.str_to_id(y)
        ids = torch.tensor(ids)
        
        return image, ids