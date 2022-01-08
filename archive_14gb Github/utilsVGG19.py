import torch
from torchvision import models
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

def set_parameter_requires_grad (model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained = True):
    model = models.vgg19_bn(pretrained = use_pretrained)
    # model.append = CovidNet
    
    set_parameter_requires_grad(model, feature_extract)
    
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    # model = models.resnet152(pretrained = use_pretrained)
    # set_parameter_requires_grad(model, feature_extract)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
     
    return model

def optimi(model_ft, device, feature_extract, lr, num_epochs):
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer = Adam(params_to_update ,lr = lr, weight_decay = lr/num_epochs)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [5,10,15], gamma=0.1, last_epoch=-1, verbose=False)
    return optimizer, scheduler