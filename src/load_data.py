import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
import numpy as np

class ProcessData():
    def load_data(self, path):
        dataset = ImageFolder(path)
        return dataset
    
    def calc_mean_and_std(self, img_ds):
        r_mean_a =[]
        g_mean_a = []
        b_mean_a=[]
        r_std_a= []
        g_std_a= []
        b_std_a= []
        tensor_t = transforms.Compose([transforms.ToTensor()])
        for (img_tr,lab) in img_ds:
            img_tr = tensor_t(img_tr)
            r_mean, g_mean, b_mean = torch.mean(img_tr, dim = [1,2])
            r_std, g_std, b_std = torch.std(img_tr, dim = [1,2])

            r_mean_a.append(r_mean.numpy())
            g_mean_a.append(g_mean.numpy())
            b_mean_a.append(b_mean.numpy())
            r_std_a.append(r_std.numpy())
            g_std_a.append(g_std.numpy())
            b_std_a.append(b_std.numpy())
        return np.mean(r_mean_a), np.mean(g_mean_a), np.mean(b_mean_a), np.mean(r_std_a), np.mean(g_std_a), np.mean(b_std_a)
    
    def apply_transformations(self):
        train_transforms = transforms.Compose([
                                transforms.Resize((224,224)),
                                               #transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
                               ])
        test_transforms = transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
                               ])
        return train_transforms, test_transforms
    
    def split_data(self, dataset):
        train_len = int(0.8*len(dataset))
        test_len =  len(dataset) - train_len
        
        train_data, test_data = torch.utils.data.random_split(dataset, lengths = [train_len, test_len])
        return train_data, test_data
    
    def create_dataloader(self, train_data, test_data, batch_size):
        train = DataLoader(train_data, batch_size = batch_size, shuffle = True)
        test = DataLoader(test_data, batch_size = batch_size, shuffle=False)
        return train, test 
    
    
          
        
        
        
        
        