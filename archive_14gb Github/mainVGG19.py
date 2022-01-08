#!/usr/bin/env python
# coding: utf-8

#importing the libraries
import pandas as pd
import numpy as np
import time

from torch.utils.data import DataLoader
import warnings
import os 
import torchvision.transforms as transforms

# from torch._C import dtype

from script.utilsVGG19 import *
from script.train import training_loop
from script.dataset import ImageDataset
from script.test import img_transform, test_loop
from script.visualize import *
# from script.data import *
#for reading and displaying images

from PIL import Image

#Pytorch libraries and modules
import torch
from torch.nn import CrossEntropyLoss

#for evaluating model
from sklearn.metrics import accuracy_score

import argparse

import matplotlib.pyplot as plt

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CHECKPOINT_PATH", default = './model/FT_VGG19_bn_cp.pt',type=str)
    parser.add_argument('--train_path', default='./dataset/train/', type=str)
    parser.add_argument('--test_path', default='./dataset/test/', type= str)

    parser.add_argument('--train_metadata', default='./dataset/train_set.txt', type=str)
    parser.add_argument('--test_metadata', default='./dataset/test_set.txt', type=str)
    parser.add_argument('--val_metadata', default='./dataset/val_set.txt', type=str)

    parser.add_argument('--BATCH_SIZE', default=32, type=int)
    parser.add_argument('--classes', default=['Negative', 'Positive'])
    parser.add_argument('--num_epochs', default= 22, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--num_classes', default=2, type=int)
    

    # parser.add_argument('--device', action='store_true', default=True)
    parser.add_argument('--feature_extract',action='store_true', default= False)
 

    opt = parser.parse_args()
    return opt



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

def dataloader():

    opt = get_opt()

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

def visualiz():

    opt = get_opt()
    # Get a batch of training data
    image, label = next(iter(dataloader()['train']))
    fig = plt.figure(figsize=(25, 7))

    # display batch_size = 40 images
    for idx in np.arange(opt.BATCH_SIZE):
        ax = fig.add_subplot(4, opt.BATCH_SIZE/4, idx+1, xticks=[], yticks=[])
        imshow(image[idx]) # lay 1 cap co nghia la o day show anh
        ax.set_title(opt.classes[label[idx]]) # vì đã chuyển từ nes/pos -> 0,1 -> tensor 0,1
    plt.show()

def predict(path_img, model_ft, verbose = False):
    if not verbose:
        warnings.filterwarnings('ignore')
    try:
        checks_if_model_is_loaded = type(model_ft)
    except:
        pass
    model_ft.eval()
    if verbose:
        print('Model loader ...')
    image = img_transform(path_img, augment()['test'])
    image1 = image[None,:,:,:]
    
    with torch.no_grad():
        outputs = model_ft(image1)
        
        _,pred_int = torch.max(outputs.data, 1)
        _,top1_idx = torch.topk(outputs.data, 1, dim = 1)
        pred_idx = int(pred_int.cpu().numpy())
        if pred_idx == 0:
            pred_str = str('Negative')
            print('img: {} is: {}'.format(os.path.basename(path_img), pred_str))
        else:
            pred_str = str('Positive')
            print('img: {} is: {}'.format(os.path.basename(path_img), pred_str))

def load_model(CHECKPOINT_PATH, model):
    checkpoint = torch.load(CHECKPOINT_PATH)#, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_list = checkpoint['loss_list']
    acc_list = checkpoint['train_acc']
    return model, loss_list, acc_list

def main():

    resnet = initialize_model(opt.num_classes, opt.feature_extract,use_pretrained=True)
    optimizer, scheduler = optimi(resnet,device, opt.feature_extract, opt.lr, opt.num_epochs)

    since = time.time()
    loss_list, acc_list = training_loop(resnet, optimizer, criterion, scheduler, device, opt.num_epochs, dataloader, opt.CHECKPOINT_PATH)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    visualize_loss(loss_list, './reportVGG19bn/lossFTVGG19bn.png')
    visualize_acc(acc_list,'./reportVGG19bn/ACCFTVGG19bn.png')

    resnet, _, _ = load_model(opt.CHECKPOINT_PATH, resnet)
    y_true, y_pred = test_loop(resnet, device, dataloader()['test'])
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)

    confusion(y_true, y_pred, opt.classes)
    report(y_true, y_pred, opt.classes, './reportVGG29bn/classification_reportVGG19bn.txt')
    
    pred_str = str('')

    path_image = './pred/covid.jpg'

    img = Image.open(path_image)
    plt.imshow(img)

    predict(path_image,resnet)
    plt.title('predict:{}'.format(pred_str))
    plt.text(5,45,'top {}:{}'.format(1,pred_str), bbox = dict(fc='yellow'))
    plt.show()

def testreport(model):

    model , loss_list, acc_list = load_model(opt.CHECKPOINT_PATH, model)

    visualize_loss(loss_list, './reportVGG19bn/lossFTVGG19bn.png')
    visualize_acc(acc_list,'./reportVGG19bn/ACCFTVGG19bn.png')
    y_true, y_pred = test_loop(model, device, dataloader()['test'])
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)

    confusion(y_true, y_pred, opt.classes)
    report(y_true, y_pred, opt.classes, './reportVGG19bn/classification_reportVGG19bn.txt')
    
    pred_str = str('')

    path_image = './pred/covid.jpg'

    img = Image.open(path_image)
    plt.imshow(img)

    predict(path_image,model)
    plt.title('predict:{}'.format(pred_str))
    plt.text(5,45,'top {}:{}'.format(1,pred_str), bbox = dict(fc='yellow'))
    plt.show()


if __name__ == '__main__':

    opt = get_opt()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = CrossEntropyLoss()
    model = initialize_model(opt.num_classes, opt.feature_extract,use_pretrained=True)
    testreport(model)
    # visualiz()
    # main()
    # dataloader()