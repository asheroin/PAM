#!/usr/bin/env python
# coding=utf-8


import numpy as np

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
# default values

ImageNetNormalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                         std = [0.229, 0.224, 0.225])


class DataSet(torch.utils.data.Dataset):
    # make a dataset iterator
    def __init__(self, file_name, transform = None, shuffle = True, split = ' '):
        """
        Args:
            file_name (string): Path to the CSV file with annotations
            transform (callable, optional): Optional transform to be applied
                on sample
        """
        self.ipt_files = [subitem.strip() for subitem in open(file_name).readlines()]
        self.tranform = transform
        self.split = split

    def __len__(self):
        return len(self.ipt_files)

    def __getitem__(self, idx):
        img_dir, img_label = self.ipt_files[idx].split(self.split)
        img_label = float(img_label)
        image = Image.open(img_dir).convert('L').convert('RGB')
        # some images may have lines less the 224
        width, height = image.size
        scale = 256
        # if width < height:
        #     new_width = scale
        #     new_height = height * scale / width
        # else:
        #     new_width = width * scale / height
        #     new_height = scale
        #  image = image.resize((new_width, new_height))
        """
        Use same settings in the PAM data
        Maybe to try scale-resize?
        """
        image = image.resize((scale, scale))
        # if set a tranform/data argumentation
        if self.tranform:
            image = self.tranform(image)
        # return value
        return (image, img_label)


class MultiTaskDataSet(torch.utils.data.Dataset):
    # make a dataset iterator
    def __init__(self, file_name, transform = None, shuffle = True, split = ' '):
        """
        Args:
            file_name (string): Path to the CSV file with annotations
            transform (callable, optional): Optional transform to be applied
                on sample
        """
        self.ipt_files = [subitem.strip() for subitem in open(file_name).readlines()]
        self.tranform = transform
        self.split = split

    def __len__(self):
        return len(self.ipt_files)

    def __getitem__(self, idx):
        img_info = self.ipt_files[idx].split(self.split)
        img_dir = img_info[0]
        img_info = img_info[1:]
        mask_len = len(img_info) / 2
        if (2 * mask_len != len(img_info)):
            print('length of inputs: {}'.format(len(img_info)))
            print(img_info)
            raise Exception,'multi marks not match'
        multi_task_mask = map(float, img_info[:mask_len])
        multi_task_score = map(float, img_info[mask_len:])
        image = Image.open(img_dir).convert('L').convert('RGB')
        # some images may have lines less the 224
        width, height = image.size
        scale = 256
        # if width < height:
        #     new_width = scale
        #     new_height = height * scale / width
        # else:
        #     new_width = width * scale / height
        #     new_height = scale
        #  image = image.resize((new_width, new_height))
        """
        Use same settings in the PAM data
        Maybe to try scale-resize?
        """
        image = image.resize((scale, scale))
        # if set a tranform/data argumentation
        if self.tranform:
            image = self.tranform(image)
        # return value
        return (image, np.array(multi_task_mask), np.array(multi_task_score))

def GetTrainLoader(traindir, batch_size, workers):
    train_loader = torch.utils.data.DataLoader(
            DataSet(traindir, transforms.Compose([
                transforms.RandomCrop(224),
                # transforms.RandomCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ImageNetNormalize,
                ])),
            batch_size = batch_size, shuffle = True,
            num_workers = workers, pin_memory = True
            )
    return train_loader


def GetValidLoader(validdir, batch_size, workers):
    valid_loader = torch.utils.data.DataLoader(
            DataSet(validdir, transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                ImageNetNormalize,
                ])),
            batch_size = batch_size, shuffle = False,
            num_workers = workers, pin_memory = True
            )
    return valid_loader



def GetMultiTaskTrainLoader(traindir, batch_size, workers):
    train_loader = torch.utils.data.DataLoader(
            MultiTaskDataSet(traindir, transforms.Compose([
                transforms.RandomCrop(224),
                # transforms.RandomCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ImageNetNormalize,
                ])),
            batch_size = batch_size, shuffle = True,
            num_workers = workers, pin_memory = True
            )
    return train_loader


def GetMultiTaskValidLoader(validdir, batch_size, workers):
    valid_loader = torch.utils.data.DataLoader(
            MultiTaskDataSet(validdir, transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                ImageNetNormalize,
                ])),
            batch_size = batch_size, shuffle = True,
            num_workers = workers, pin_memory = True
            )
    return valid_loader


