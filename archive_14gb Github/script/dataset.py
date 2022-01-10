#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torchvision.transforms as transforms
from torchvision.transforms.functional import scale
from torchvision.transforms.transforms import CenterCrop, Normalize, RandomHorizontalFlip, ToTensor

def augment():
    data_transforms = {
        'train' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees = 0, shear = 0.2),    
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = np.array([0.229, 0.224, 0.225])),
        ]),
        'test' : transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = np.array([0.229, 0.224, 0.225]))
        ])
    }
    return data_transforms

def augmentCovid():
    data_transforms = {
        'train' : transforms.Compose([            
            transforms.RandomAffine(degrees = 10, translate=(0.1,0.1), fill =0,shear = 0.2),    
            transforms.CenterCrop(10),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.9, 1.1)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = np.array([0.229, 0.224, 0.225])),
        ]),
        'test' : transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = np.array([0.229, 0.224, 0.225]))
        ])
    }
    return data_transforms

def augmentCovidpytoch():
    data_transforms = {
        'train' : transforms.Compose([            
            transforms.Resize(256),
            transforms.RandomResizedCrop((224), scale = (0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]),
        'test' : transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = ([0.229, 0.224, 0.225]))
        ])
}
    return data_transforms
    
def dataloader(opt):

    # opt = get_opt()

    train_txt= pd.read_csv(opt.train_metadata, sep= '\s+', header=None)
    test_txt = pd.read_csv(opt.test_metadata, sep= '\s+', header=None)
    val_txt = pd.read_csv(opt.val_metadata,sep = "\s+", header= None)
    

    train_txt.columns = ["file_name","label"]
    test_txt.columns = ["file_name","label"]
    val_txt.columns = ["file_name","label"]

    # print(train_txt.count())
    # print(val_txt['label'].value_counts())
    # print(val_txt.count())

    train_dataset = ImageDataset(train_txt,opt.train_path,augment()['train'])
    test_dataset = ImageDataset(test_txt,opt.test_path,augment()['test'])
    val_dataset = ImageDataset(val_txt,opt.train_path,augment()['test'])

    # print(train_dataset)

    loader ={
        'train' : DataLoader(
            train_dataset, 
            batch_size= opt.BATCH_SIZE,
            shuffle=True
        ),
        'val' : DataLoader(
            val_dataset, 
            batch_size=opt.BATCH_SIZE,
            shuffle=True
        ),
        'test' : DataLoader(
            test_dataset, 
            batch_size=opt.BATCH_SIZE,
            shuffle=True
        )
    }   
    return loader

class ImageDataset(Dataset):
    def __init__(self,csv,img_folder,transform): # 'Initialization'
        self.csv=csv
        self.transform=transform
        self.img_folder=img_folder
    
        self.image_names=self.csv[:]['file_name']# [:] lấy hết số cột số hàng của bảng
        self.labels= np.array(self.csv[:]['label']) # note kiểu mảng int đúng không?
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.image_names)

    def __getitem__(self,index): # 'Generates one sample of data'
    
        image=Image.open(self.img_folder + self.image_names.iloc[index]).convert('RGB')
#         print('image',image)
        image=self.transform(image)
        targets=self.labels[index]
        targets = torch.tensor(targets, dtype=torch.long) #đọc từng phần tử của mảng, chuyển từ array -> tensor; kiểu int64 tương ứng với long trong pytorch

        return image, targets # chua 1 cap
#   transforms.Resize((224, 224))(image)
            #   t = transforms.RandomAffine(degrees = 0, shear = 0.2)     , t(image) 
#   tr = 
