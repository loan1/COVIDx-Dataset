#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

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
