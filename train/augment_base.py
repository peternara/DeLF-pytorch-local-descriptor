# -*- coding: utf-8 -*- 

import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

#unnormal_tranform  = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#images = unnormal_tranform(images)

def augmentation_base(network, aug_type='train'):

    if network=='vgg16' or 'efficientnet' in network or \
         'resnet' in network or 'resnext' in network or 'mobilenet' in network:

       TRAIN_IMAGE_SIZE = [256, 256]
       IMAGE_SIZE       = [224, 224]

       if aug_type=='train':
          print('augment - train')
          image_transform = transforms.Compose([
                             transforms.ToPILImage(), # 
                             transforms.Resize(256),
                             transforms.RandomAffine(degrees=360, scale=(0.7, 1.3), shear=30),
                             #transforms.RandomAffine(degrees=360, shear=30),
                             #transforms.RandomAffine(degrees=360),
                             #transforms.RandomCrop(224),                              
                             transforms.RandomResizedCrop(224, scale=(0.4, 1)),
                             #transforms.ColorJitter(brightness=0.4, contrast=0.4),
                             transforms.RandomHorizontalFlip(),
                             #transforms.RandomVerticalFlip(),
                             transforms.ToTensor(), # HxWxC:[0, 1]> CxHxW:[0,255]
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
       elif aug_type=='train_attention':
          print('augment - train_attention')
          image_transform = transforms.Compose([
                             transforms.ToPILImage(), # 
                             transforms.Resize(900),
                             transforms.RandomAffine(degrees=360, scale=(0.7, 1.3), shear=30),
                             #transforms.RandomAffine(degrees=360, shear=30),
                             #transforms.RandomAffine(degrees=360),
                             #transforms.RandomCrop(224),                              
                             transforms.RandomResizedCrop(720, scale=(0.4, 1)),
                             #transforms.ColorJitter(brightness=0.4, contrast=0.4),
                             transforms.RandomHorizontalFlip(),
                             #transforms.RandomVerticalFlip(),
                             transforms.ToTensor(), # HxWxC:[0, 1]> CxHxW:[0,255]
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])                 
       elif aug_type=='val':
          print('augment - val')
          image_transform = transforms.Compose([
                             transforms.ToPILImage(), # 
                             #transforms.Resize(224),
                             #transforms.CenterCrop(IMAGE_SIZE[0]),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
       else: # test
          print('augment - test') 
          # 매칭할때..
          image_transform = transforms.Compose([
                             transforms.ToPILImage(), # 
                             #transforms.Resize(224), # not working ??
                             #transforms.CenterCrop(IMAGE_SIZE[0]),
                             #transforms.Resize(int(IMAGE_SIZE[0]/0.875)),
                             #transforms.CenterCrop(IMAGE_SIZE[0]),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

          # classification 할때 조건..
          ''' 
          image_transform = transforms.Compose([
                             transforms.ToPILImage(), # 
                             #transforms.Resize(224), # not working ??
                             #transforms.CenterCrop(IMAGE_SIZE[0]),
                             transforms.Resize(int(IMAGE_SIZE[0]/0.875)),
                             transforms.CenterCrop(IMAGE_SIZE[0]),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
           '''                           
       return image_transform






