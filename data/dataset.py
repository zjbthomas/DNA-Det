import random
import numpy as np

import cv2

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.common import read_annotations
from data.transforms import MultiCropTransform, get_transforms

class ImageDataset(Dataset):
    def __init__(self, annotations, config, opt, balance=False):
        self.opt = opt
        self.config = config
        self.balance = balance
        self.class_num=config.class_num
        self.resize_size = config.resize_size
        self.second_resize_size = config.second_resize_size
        self.norm_transform = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
        if balance:
            self.data = [[x for x in annotations if x[1] == lab] for lab in [i for i in range(self.class_num)]]
        else:
            self.data = [annotations]

    def __len__(self):
        
        return max([len(subset) for subset in self.data])

    def __getitem__(self, index):
        
        if self.balance:
            labs = []
            imgs = []
            img_paths = []
            for i in range(self.class_num):
                safe_idx = index % len(self.data[i])
                img_path, lab = self.data[i][safe_idx]
                img = self.load_sample(img_path)
                labs.append(lab)
                imgs.append(img)
                img_paths.append(img_path)
                
            return torch.cat([imgs[i].unsqueeze(0) for i in range(self.class_num)]),\
                torch.tensor(labs, dtype=torch.long), img_paths
        else:
            img_path, lab = self.data[0][index]
            img = self.load_sample(img_path)
            lab = torch.tensor(lab, dtype=torch.long)
            
            return img, lab, img_path

    def load_sample(self, img_path):
        transform_crop = A.Compose([
            A.CenterCrop(self.config.crop_size[0], self.config.crop_size[1])
        ])

        if self.resize_size is not None:
            transform_resize = A.Compose([
                A.Resize(self.resize_size[0], self.resize_size[1])
            ])

        if self.second_resize_size is not None:
            transform_second_resize = A.Compose([
                A.Resize(self.second_resize_size[0], self.second_resize_size[1])
            ])


        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        height, width, _ = img.shape
        if height != width:
            img = transform_crop(image = img)['image']
        if self.resize_size is not None:
            img = transform_resize(image = img)['image']
        if self.second_resize_size is not None:
            img = transform_second_resize(image = img)['image']
        
        img = self.norm_transform(image = img)['image']

        return img


class ImageMultiCropDataset(ImageDataset):
    def __init__(self, annotations, config, opt, balance=False):
        super(ImageMultiCropDataset, self).__init__(annotations, config, opt, balance)
        
        self.multi_size = config.multi_size

        crop_transforms = []
        for s in self.multi_size:
            transform_crop = A.Compose([
                A.RandomCrop(s[0], s[1])
            ])
            crop_transforms.append(transform_crop)
            self.multicroptransform = MultiCropTransform(crop_transforms)
            
    def __getitem__(self, index):
        
        if self.balance:
            labs = []
            imgs = []
            crops = []
            img_paths = []
            for i in range(self.class_num):
                safe_idx = index % len(self.data[i])
                img_path = self.data[i][safe_idx][0]
                img, crop = self.load_sample(img_path)
                lab = self.data[i][safe_idx][1]
                labs.append(lab)
                imgs.append(img)
                crops.append(crop)
                img_paths.append(img_path)
            crops = [torch.cat([crops[c][size].unsqueeze(0) for c in range(self.class_num)])
                for size in range(len(self.multi_size))]

            return torch.cat([imgs[i].unsqueeze(0) for i in range(self.class_num)]),\
                crops, torch.tensor(labs, dtype=torch.long), img_paths
        else:
            img_path, lab = self.data[0][index]
            lab = torch.tensor(lab, dtype=torch.long)
            img, crops = self.load_sample(img_path)

            return img, crops, lab, img_path

    def load_sample(self, img_path):
        transform_crop = A.Compose([
            A.CenterCrop(self.config.crop_size[0], self.config.crop_size[1])
        ])

        if self.resize_size is not None:
            transform_resize = A.Compose([
                A.Resize(self.resize_size[0], self.resize_size[1])
            ])

        if self.second_resize_size is not None:
            transform_second_resize = A.Compose([
                A.Resize(self.second_resize_size[0], self.second_resize_size[1])
            ])


        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape
        if height != width:
            img = transform_crop(image = img)['image']
        if self.resize_size is not None:
            img = transform_resize(image = img)['image']
        if self.second_resize_size is not None:
            img = transform_second_resize(image = img)['image']

        crops = self.multicroptransform(img)
        img = self.norm_transform(image = img)['image']
        crops = [self.norm_transform(image = crop)['image'] for crop in crops]

        return img, crops

class ImageTransformationDataset(ImageDataset):
    def __init__(self, annotations, config, opt, balance=False):
        super(ImageTransformationDataset, self).__init__(annotations, config, opt, balance)
    
        self.data = annotations
        self.pretrain_transforms = get_transforms(config.crop_size)
        self.class_num = self.pretrain_transforms.class_num
        crop_transforms = []
        self.multi_size = config.multi_size
        for s in self.multi_size:
            transform_crop = A.Compose([
                A.RandomCrop(s[0], s[1])
            ])
            crop_transforms.append(transform_crop)
            self.multicroptransform = MultiCropTransform(crop_transforms)
            
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, index):
        transform_crop = A.Compose([
            A.RandomCrop(self.config.crop_size[0], self.config.crop_size[1])
        ])

        img_path = self.data[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        img = transform_crop(image = img)['image']
        
        select_id=random.randint(0,self.class_num-1)
        pretrain_transform=self.pretrain_transforms.select_tranform(select_id)
        transformed = pretrain_transform(image=np.asarray(img))
        img = cv2.cv.fromarray(transformed["image"])

        if self.resize_size is not None:
            img = img.resize(self.resize_size)

        crops = self.multicroptransform(img)
        img = self.norm_transform(image = img)['image']
        crops = [self.norm_transform(image = crop)['image'] for crop in crops]
        lab = torch.tensor(select_id, dtype=torch.long)
    
        return img, crops, lab, img_path    

class BaseData(object):
    def __init__(self, train_data_path, val_data_path, config, opt):

        train_set = ImageDataset(read_annotations(train_data_path), config, opt, balance=True)
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
        
        val_set = ImageDataset(read_annotations(val_data_path), config, opt, balance=False)
        val_loader = DataLoader(
            dataset=val_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        print('train: {}, val: {}'.format(len(train_set),len(val_set)))


class SupConData(object):
    def __init__(self, train_data_path, val_data_path, config, opt):
        
        train_set = ImageMultiCropDataset(read_annotations(train_data_path), config, opt, balance=True)
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

        val_set = ImageMultiCropDataset(read_annotations(val_data_path), config, opt, balance=False)
        val_loader = DataLoader(
            dataset=val_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        print('train: {}, val: {}'.format(len(train_set),len(val_set)))


class TranformData(object):
    def __init__(self, train_data_path, val_data_path, config, opt):

        
        train_set = ImageTransformationDataset(read_annotations(train_data_path), config, opt)
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
        self.train_loader = train_loader
        self.class_num = train_set.class_num

        val_set = ImageTransformationDataset(read_annotations(val_data_path), config, opt)
        val_loader = DataLoader(
            dataset=val_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
        self.val_loader = val_loader
        
        print('train: {}, val: {}'.format(len(train_set),len(val_set)))
        

