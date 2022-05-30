import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image
import cv2 as cv
import numpy as np

class CaptchaDataset(Dataset):
    def __init__(self, path, isSelection=False, transform=transforms.ToTensor(), isGrayscale=False, isCrop=False, isFilter=False):
        self.transform = transform
        self.isCrop = isCrop
        self.isFilter = isFilter
        self.isSelection = isSelection
        self.x, self.y = self.loadlist(path)
    
    def loadlist(self, path):
        x = []
        y = []
        for filename in glob.glob(path+'/*'):
            x.append(filename)
            y_str = filename.split('/')[-1].split('.')[0]
            if self.isSelection:
                ids = self.str_to_selection(y_str)
            else:
                ids = self.str_to_id(y_str)
            y.append(ids)
            
        
        return x, y
    
    def str_to_id(self, y):
        keys = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        dic = {k:i for i, k in enumerate(keys)}
        ids = list(map(lambda x: dic[x], y))
        return ids
    
    def str_to_selection(self, y):
        keys = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        dic = {k:i for i, k in enumerate(keys)}
        ids = list(map(lambda x: dic[x], y))
        result = [0 for i in range(len(keys))]
        for _y in y:
            result[dic[_y]]=1
        return result
        
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        img = Image.open(x)
        image = np.array(img)
        if self.isCrop:
            image = image[85:170,:,:]
        if self.isFilter:
            image = cv.medianBlur(image, 5)
        img = Image.fromarray(image)
        image = self.transform(img)
        img.close()
        
        y = self.y[idx]
        # if self.isSelection:
        #     ids = self.str_to_selection(y)
        # else:
        #     ids = self.str_to_id(y)
        y = torch.tensor(y)
        
        return image, y