
# -*- coding: utf-8 -*- 

import os
import cv2
# import time
import numpy as np
# import imgaug as ia
# import imgaug.augmenters as iaa

import torch.utils.data as data
from collections import Counter

#IMAGE_SHAPE = (300, 300)
IMAGE_SHAPE = (224, 224)
IMAGE_SHAPE2 = (256, 256)

SEED = 20190519
EVAL_RATIO = 0.05


class ListLoader(object):
    def __init__(self, root_path, file_name=None, label_to_cls=None, type=0):
        np.random.seed(SEED)

        self.category_count = Counter()  # number of images for each category
        self.image_list     = []
        self.labelmap       = {}
        self.labelmap_names = {}

        if label_to_cls==None:
           self.label_to_cls   = {} # label_to_cls
        else:
           self.label_to_cls   = label_to_cls # train set에서 eval에 전달..

        idx = 0
        if type==0: # diretory type

           # self.import_labelmap 함수를 대체한다.
           dir_to_labels = dict()
           label_names   = dict()
           for directory in os.walk(root_path):
               for dir_name in directory[1]:  # All subdirectories > ['1', '0', '2']
                   label_name = 'label_'+str(dir_name)
                   dir_to_labels[dir_name] = dir_name
                   label_names[dir_name]   = label_name 

           # 0       wallet
           #dir_to_labels, label_names = self.import_labelmap(os.path.join(root_path, 'category.list')) 

           idx = 0
           for directory in os.walk(root_path):
               for dir_name in directory[1]:  # All subdirectories > ['1', '0', '2']
                   type_id    = idx
                   type_name  = dir_to_labels[dir_name]
                   label_name = label_names[dir_name]
                   # print(dir_name, label_name, type_id)

                   self.labelmap[type_id]       = type_name                
                   self.labelmap_names[type_id] = label_name

                   for image_file in os.listdir(os.path.join(root_path, dir_name)):
                       self.category_count[type_id] += 1
                
                   #if  self.category_count[type_id] < 10 :  # 15
                   #    continue

                   for image_file in os.listdir(os.path.join(root_path, dir_name)):
                       full_path = os.path.join(root_path, dir_name, image_file)
                       self.image_list.append((full_path, type_id))

                   self.label_to_cls[label_name]  = type_id

                   idx+=1
        elif type==1 : # file type

           label_images = {}
           train_path   = os.path.join(root_path, file_name)
          
           # 전체 클래스 갯수.
           for line in open(train_path, "r"):
               label, img_path = line.strip().split(',') # neuralcode_0,DB/Instance/neuralcode/DIR/landmarks_clean_train/000/06789.jpg
               if label not in label_images:
                  label_images[label] = []
               label_images[label].append(os.path.join(root_path, img_path)) # 사실 fullpath가 저장되어야하는 nsml 정책

           idx = 0
           for label in label_images:
               type_id    = idx
               if label_to_cls!=None: # trainset에서 가져온걸 eval set에 적용한다.
                  if label not in self.label_to_cls :
                     continue
                  type_id = self.label_to_cls[label]

               if len(label_images[label])<15:
                  continue

               self.category_count[type_id] = len(label_images[label])
               self.labelmap[type_id]       = label                           
               self.labelmap_names[type_id] = label
  
               for full_path in label_images[label]:
                   self.image_list.append((full_path, type_id))

               if label_to_cls==None:
                  self.label_to_cls[label]     = type_id
               idx+=1

        self.num_classes = idx
        print('num_classes : ' , self.num_classes)
        print('all image :  ', sum(self.category_count.values()), len(self.category_count))
        avg_count = sum(self.category_count.values()) / len(self.category_count)
        print('Avg count per category:', avg_count)
        minimum = min(self.category_count, key=self.category_count.get)
        print('Min count category:', self.category_count[minimum])
        maximum = max(self.category_count, key=self.category_count.get)
        print('Max count category:', self.category_count[maximum])

        self.category_multiple = {}
        small_cat = 0
        for type_id in self.category_count:
            multiple = int(3 * avg_count / self.category_count[type_id])
            if multiple > 1:
                small_cat += 1
            self.category_multiple[type_id] = multiple
        print('Small categories:', small_cat)

    def image_indices_all(self, oversample=True):
        '''Return train/eval image files' list'''
        length        = len(self.image_list)
        indices       = np.random.permutation(length)
        train_indices = indices

        if oversample:
           # For categories which have small number of images, oversample it
           extra_train_indices = []
           for index in train_indices:
               _, type_id = self.image_list[index]
               multiple   = self.category_multiple[type_id]

               if multiple > 1:
                  for i in range(multiple):
                      extra_train_indices.append(index)

           extra_train_indices = np.asarray(extra_train_indices, dtype=train_indices.dtype)
           train_indices       = np.concatenate((train_indices, extra_train_indices))
          

        return self.image_list, train_indices

    def image_indices(self):
        '''Return train/eval image files' list'''
        length        = len(self.image_list)
        indices       = np.random.permutation(length)
        point         = int(length * EVAL_RATIO)
        eval_indices  = indices[0:point]
        train_indices = indices[point:]

        # For categories which have small number of images, oversample it
        extra_train_indices = []
        for index in train_indices:
            _, type_id = self.image_list[index]
            multiple   = self.category_multiple[type_id]

            if multiple > 1:
                for i in range(multiple):
                    extra_train_indices.append(index)

        extra_train_indices = np.asarray(extra_train_indices, dtype=train_indices.dtype)
        train_indices       = np.concatenate((train_indices, extra_train_indices))

        return self.image_list, train_indices, eval_indices

    def multiples(self):
        return self.category_multiple

    def import_labelmap(self, path):
        file          = open(path, 'r')
        dir_to_labels = dict()
        label_names   = dict()
        for line in file.readlines():
            line                    = line.strip().split('\t') # 0	cls_name
            dir_name                = line[0].strip()
            label_name              = line[1].strip()
            dir_to_labels[dir_name] = dir_name
            label_names[dir_name]   = label_name
        file.close()
        return dir_to_labels, label_names

    def export_labelmap(self, path='labelmap.csv'):
        with open(path, 'w') as fp:
            for type_id, type_name in self.labelmap.items(): # type_id=cls, type_name=cls_dirname, 
                count      = self.category_count[type_id]
                label_name = self.labelmap_names[type_id]
                fp.write(str(type_id) + ',' + type_name + ',' + str(count) + ','+label_name + '\n')


class PulsClassificationDataset(data.Dataset):
    """ All images and classes for birds through the world """
    def __init__(self, image_list, image_indices, category_multiple, transform=None, data_type=None):
        self.image_list        = image_list
        self.image_indices     = image_indices
        self.category_multiple = category_multiple
        self.transform         = transform
        self.data_type         = data_type

    def __getitem__(self, index):
        image_path, type_id = self.image_list[self.image_indices[index]]
        image               = cv2.imread(image_path)

        if image is None:
            print("[Error] {} can't be read".format(image_path))
            return None

        if self.data_type=='train' and image.shape != (IMAGE_SHAPE2[0], IMAGE_SHAPE2[1], 3):
            image = cv2.resize(image, IMAGE_SHAPE2) # resize : IMAGE_SHAPE2 for(-> Random Crop -> IMAGE_SHAPE)
        elif self.data_type=='val' and image.shape != (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3):
            image = cv2.resize(image, IMAGE_SHAPE)

        if image is None:
            print("[Error] {} can't be read".format(image_path))
            return None

        if self.transform:
            image = self.transform(image) 

        return image, int(type_id), image_path

    def __len__(self):
        return len(self.image_indices)

    @staticmethod
    def my_collate(batch):
        batch = filter(lambda img: img is not None, batch)
        return data.dataloader.default_collate(list(batch))

