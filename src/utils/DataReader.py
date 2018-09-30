#!/usr/bin/env python
# coding=utf-8


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
        if width < height:
            new_width = 255
            new_height = height * 255 / width
        else:
            new_width = width * 255 / height
            new_height = 255
        image = image.resize((new_width, new_height))
        # if set a tranform/data argumentation
        if self.tranform:
            image = self.tranform(image)
        # return value
        return (image, img_label)



def GetTrainLoader(traindir, batch_size, workers):
    train_loader = torch.utils.data.DataLoader(
            DataSet(traindir, transforms.Compose([
                transforms.RandomCrop(224),
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
            batch_size = batch_size, shuffle = True,
            num_workers = workers, pin_memory = True
            )
    return valid_loader





