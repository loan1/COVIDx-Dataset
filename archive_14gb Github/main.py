#!/usr/bin/env python
# coding: utf-8

#importing the libraries
import pandas as pd
import numpy as np
import time

# from torch._C import dtype

from script.utils import *
from script.train import training_loop
from script.dataset import ImageDataset
from script.test import img_transform, test_loop
from script.visualize import *
from script.dataset import augment, dataloader
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
    parser.add_argument("--CHECKPOINT_PATH", default = './model/CovidNet.pt',type=str)
    parser.add_argument('--train_path', default='./dataset/train/', type=str)
    parser.add_argument('--test_path', default='./dataset/test/', type= str)

    parser.add_argument('--train_metadata', default='./dataset/train_set.txt', type=str)
    parser.add_argument('--test_metadata', default='./dataset/test_set.txt', type=str)
    parser.add_argument('--val_metadata', default='./dataset/val_set.txt', type=str)

    parser.add_argument('--BATCH_SIZE', default=32, type=int)
    parser.add_argument('--classes', default=['Negative', 'Positive'])
    parser.add_argument('--num_epochs', default= 22, type=int)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--num_classes', default=2, type=int)
    

    # parser.add_argument('--device', action='store_true', default=True)
    parser.add_argument('--feature_extract',action='store_true', default= False)
 

    opt = parser.parse_args()
    return opt

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
def visualiz():

    opt = get_opt()
    # Get a batch of training data
    image, label = next(iter(dataloader(opt)['train']))
    fig = plt.figure(figsize=(25, 7))

    # display batch_size = 40 images
    for idx in np.arange(opt.BATCH_SIZE):
        ax = fig.add_subplot(4, opt.BATCH_SIZE/4, idx+1, xticks=[], yticks=[])
        imshow(image[idx]) # lay 1 cap co nghia la o day show anh
        ax.set_title(opt.classes[label[idx]]) # vì đã chuyển từ nes/pos -> 0,1 -> tensor 0,1
    plt.show()



def load_train_continue(CHECKPOINT_PATH, model):
    checkpoint = torch.load(CHECKPOINT_PATH)#, map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    model, loss_list, acc_list = load_model(CHECKPOINT_PATH, model)

    loss_list, acc_list = training_loop(
        model, optimizer, criterion, scheduler, device, opt.num_epochs, dataloader(opt), opt.CHECKPOINT_PATH, epoch, loss_list, acc_list)
    return model, loss_list, acc_list

def load_model(CHECKPOINT_PATH, model):
    checkpoint = torch.load(CHECKPOINT_PATH)#, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_list = checkpoint['loss_list']
    acc_list = checkpoint['train_acc']

    return model, loss_list, acc_list

def main(model, first_train = True):

    # model = initialize_model(opt.num_classes, opt.feature_extract,use_pretrained=True)
    # optimizer, scheduler = optimi(model,device, opt.feature_extract, opt.lr, opt.num_epochs)

    since = time.time()

    if first_train == True:
        loss_list, acc_list = training_loop(model, optimizer, criterion, scheduler, device, opt.num_epochs, dataloader(opt), opt.CHECKPOINT_PATH)
    else:
        model, loss_list, acc_list = load_train_continue(opt.CHECKPOINT_PATH, model)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    visualize_loss(loss_list, './reportCOVIDNetAUG/loss_COVIDNet.png')
    visualize_acc(acc_list,'./reportCOVIDNetAUG/ACC_COVIDNet.png')

    # model, _, _ = load_train_continue(opt.CHECKPOINT_PATH, model)
    y_true, y_pred = test_loop(model, device, dataloader(opt)['test'])
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)

    confusion(y_true, y_pred, opt.classes)
    ##################################################

    report(y_true, y_pred, opt.classes, './reportCOVIDNetAUG/classification_reportCOVIDNet.txt')
    
    pred_str = str('')

    path_image = './pred/covid.jpg'

    img = Image.open(path_image)
    plt.imshow(img)

    predict(path_image,model)
    plt.title('predict:{}'.format(pred_str))
    plt.text(5,45,'top {}:{}'.format(1,pred_str), bbox = dict(fc='yellow'))
    plt.show()

def testreport(model):

    model , loss_list, acc_list = load_model(opt.CHECKPOINT_PATH, model)

    visualize_loss(loss_list, './reportCOVIDNetAUG/lossCOVIDNet.png')
    visualize_acc(acc_list,'./reportCOVIDNetAUG/ACCCOVIDNet.png')
    y_true, y_pred = test_loop(model, device, dataloader()['test'])

    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)

    confusion(y_true, y_pred, opt.classes)
    report(y_true, y_pred, opt.classes, './reportCOVIDNetAUG/classification_reportCOVIDNet.txt')
    
    pred_str = str('')

    path_image = './pred/covid.jpg'

    img = Image.open(path_image)
    plt.imshow(img)

    predict(path_image,model)
    plt.title('predict:{}'.format(pred_str))
    plt.text(5,45,'top {}:{}'.format(1,pred_str), bbox = dict(fc='yellow'))
    plt.show()

def reportevalution(model):

    model , loss_list, acc_list = load_model(opt.CHECKPOINT_PATH, model)
    # model , loss_list, acc_list = load_model('./model/covidnet.pt', model)
    
    # print(loss_list)
    y_true, y_pred = test_loop(model, device, dataloader(opt)['test'])
    
    report(y_true, y_pred, opt.classes, './reportCOVIDNetAUG/classification_reportCOVIDNet.txt')

    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    TN = confusion[0,0]
    FP = confusion[0,1]
    FN = confusion[1,0]
    TP = confusion[1,1]
    # print(TP, FN, FP, TN)
    
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1_score = (2*TP)/(2*TP + FP+FN)
    accuracy = (TP+TN)/(TP+TN+FP+FN)

    specificity_P = TN/(TN+FP)
    specificity_N = TP/(TP + FN)
    specificity = (specificity_P + specificity_N)/2
    print(precision, recall, f1_score, accuracy, specificity)

if __name__ == '__main__':

    opt = get_opt()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = CrossEntropyLoss()
    model = initialize_model(opt.num_classes, opt.feature_extract,use_pretrained=True)
    optimizer, scheduler = optimi(model,device, opt.feature_extract, opt.lr, opt.num_epochs)
    # testreport(model)
    # visualiz()
    main(model, first_train=True)
    reportevalution(model)
    # dataloader()
    # model , loss_list, acc_list = load_model('./model/covidnet.pt', model)
    # print(acc_list)
