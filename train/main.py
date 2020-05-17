'''
main.py
'''

import os, sys, time
sys.path.append('../')
import shutil
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from classification_dataset import PulsClassificationDataset, ListLoader, IMAGE_SHAPE
from augment_base import *

from config import config


def main():
    # print config.
    state = {k: v for k, v in config._get_kwargs()}
    print(state)

    # if use cuda.
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    use_cuda = torch.cuda.is_available()

    # Random seed
    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(config.manualSeed)
        torch.backends.cudnn.benchmark = True           # speed up training.
    
    # data loader
    from dataloader import get_loader
    if config.stage in ['finetune']:
        sample_size = config.finetune_sample_size
        crop_size   = config.finetune_crop_size
    elif config.stage in ['keypoint']:
        sample_size = config.keypoint_sample_size
        crop_size   = config.keypoint_crop_size
   
    '''
    # dataloader for pretrain
    train_loader_pt, val_loader_pt = get_loader(
        train_path       = config.train_path_for_pretraining,
        val_path         = config.val_path_for_pretraining,
        stage            = config.stage,
        train_batch_size = config.train_batch_size,
        val_batch_size   = config.val_batch_size,
        sample_size      = sample_size,
        crop_size        = crop_size,
        workers          = config.workers)

    # dataloader for finetune
    train_loader_ft, val_loader_ft = get_loader(
        train_path       = config.train_path_for_finetuning,
        val_path         = config.val_path_for_finetuning,
        stage            = config.stage,
        train_batch_size = config.train_batch_size,
        val_batch_size   = config.val_batch_size,
        sample_size      = sample_size,
        crop_size        = crop_size,
        workers          = config.workers)
    '''

    dataset_root = '/content/drive/My Drive/DataSet/Landmark/landmarks_clean_train/'
    dataset_type = 0
    if dataset_type==0:   list_loader = ListLoader( dataset_root, type=dataset_type) 
    elif dataset_type==1: list_loader = ListLoader( dataset_root, file_name=args.dataset_path, type=dataset_type)

    num_classes = list_loader.num_classes 
    config.ncls = num_classes

    num_images  = sum(list_loader.category_count.values())
    print ('num_classes : ', num_classes)
    print ('image_size  : ', num_images)

    label_output = os.path.join( dataset_root, 'labelmap.csv')    
    print ('classifcation label Info > labelmap.csv : ', label_output)
    list_loader.export_labelmap(path=label_output)    
    image_list, train_indices, eval_indices = list_loader.image_indices()

    backbone = 'resnet50'
    train_image_tranform = augmentation_base(backbone, aug_type='train')
    test_image_tranform  = augmentation_base(backbone, aug_type='val')
     
    train_set = PulsClassificationDataset(image_list, train_indices, list_loader.multiples(), transform=train_image_tranform, data_type='train')
    eval_set  = PulsClassificationDataset(image_list, eval_indices, list_loader.multiples(), transform=test_image_tranform, data_type='val') 
    print('train set: {} vs eval set: {}'.format(len(train_set), len(eval_set)))

    train_loader_pt = data.DataLoader(train_set, config.train_batch_size, num_workers=config.workers,
                                   shuffle=True, pin_memory=True,
                                   collate_fn=PulsClassificationDataset.my_collate)
    val_loader_pt   = data.DataLoader(eval_set,config.val_batch_size, num_workers=config.workers,
                                   shuffle=False, pin_memory=True,
                                   collate_fn=PulsClassificationDataset.my_collate)


    # load model
    from delf import Delf_V1
    model = Delf_V1(
        ncls                     = config.ncls,
        load_from                = config.load_from,
        arch                     = config.arch,
        stage                    = config.stage,
        target_layer             = config.target_layer,
        use_random_gamma_rescale = config.use_random_gamma_rescale)

    # solver
    from solver import Solver
    solver = Solver(config=config, model=model)
    if config.stage in ['finetune']:
        epochs = config.finetune_epoch
    elif config.stage in ['keypoint']:
        epochs = config.keypoint_epoch

    # train/test for N-epochs. (50%: pretain with datasetA, 50%: finetune with datasetB)
    config.train_path_for_pretraining = dataset_root
    
    for epoch in range(epochs):
        if epoch < int(epochs * 0.5):
            print('[{:.1f}] load pretrain dataset: {}'.format(
                float(epoch) / epochs,
                config.train_path_for_pretraining))
            train_loader = train_loader_pt
            val_loader   = val_loader_pt
        else:
            print('[{:.1f}] load finetune dataset: {}'.format(
                float(epoch) / epochs,
                config.train_path_for_finetuning))
            train_loader = train_loader_ft
            val_loader   = val_loader_ft

        solver.train('train', epoch, train_loader, val_loader)
        solver.train('val', epoch, train_loader, val_loader)

    print('Congrats! You just finished DeLF training.')


if __name__ == '__main__':
    main()
