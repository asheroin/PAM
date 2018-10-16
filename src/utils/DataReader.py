#!/usr/bin/env python
# coding=utf-8


import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.transforms import functional as tf_func
import torchvision.datasets as datasets


import numpy as np
from PIL import Image

# data preprocess settings
# be careful when using different pre-trained models

is_ToBGR = True
is_To255Tensor = True

########################### interface of transform funciton ###########################

class To255Tensor(object):

    def __init__(self, is_to_255):
        self.is_to_255 = is_to_255

    def __call__(self, pic):
        # original settings
        if not self.is_to_255:
            return tf_func.to_tensor(pic)
        # to a 255 tensor
        if not isinstance(pic, Image.Image):
            raise Exception("Only for PIL Image")
        if pic.mode != "RGB":
            raise Exception("Only for RGB Image")
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        nchannel = 3
        # rebuild from buffer to tensor
        # PIL.size = width, height
        # numpy format:[height, width, nchannel]
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # from HWC to CHW
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float()

class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor

class ToNormalizedData(object):

    def __init__(self, is_to_255):
        self.is_to_255 = is_to_255
        self.float_mean = [0.485, 0.456, 0.406]
        self.float_std = [0.229, 0.224, 0.225]
        self.uint8_mean = [104, 117, 128]
        self.uint8_std = [1, 1, 1]

    def __call__(self, tensor):
        # select normalized parameters
        if self.is_to_255:
            mean = self.uint8_mean
            std = self.uint8_std
        else:
            mean = self.float_mean
            std = self.float_std
        # see torchvision/transforms/functional.py#159
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

########################### preprocess method list  ###########################

## for train
tf_list_train = [
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            To255Tensor(is_To255Tensor),
            ToSpaceBGR(is_ToBGR),
            ToNormalizedData(is_To255Tensor)
        ]
## for validation/test
tf_list_valid = [
            # transforms.Scale([224,224]),
            transforms.CenterCrop(224),
            To255Tensor(is_To255Tensor),
            ToSpaceBGR(is_ToBGR),
            ToNormalizedData(is_To255Tensor)
        ]

########################### get iterable object ###########################

class DataSet(torch.utils.data.Dataset):

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
        # using PIL, image channels is RGB
        # However, pretrain-model is BGR
        image = Image.open(img_dir).convert('RGB')
        # image = Image.open(img_dir)
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
        image = Image.open(img_dir).convert('RGB')
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


########################### return data loader ###########################

def GetTrainLoader(traindir, batch_size, workers):
    train_loader = torch.utils.data.DataLoader(
            DataSet(traindir,
                    transforms.Compose(tf_list_train)),
            batch_size = batch_size, shuffle = True,
            num_workers = workers, pin_memory = True
            )
    return train_loader


def GetValidLoader(validdir, batch_size, workers):
    valid_loader = torch.utils.data.DataLoader(
            DataSet(validdir,
                    transforms.Compose(tf_list_valid)),
            batch_size = batch_size, shuffle = False,
            num_workers = workers, pin_memory = True
            )
    return valid_loader



def GetMultiTaskTrainLoader(traindir, batch_size, workers):
    train_loader = torch.utils.data.DataLoader(
            MultiTaskDataSet(traindir,
                             transforms.Compose(tf_list_train)),
            batch_size = batch_size, shuffle = True,
            num_workers = workers, pin_memory = True
            )
    return train_loader


def GetMultiTaskValidLoader(validdir, batch_size, workers):
    valid_loader = torch.utils.data.DataLoader(
            MultiTaskDataSet(validdir,
                            transforms.Compose(tf_list_valid )),
            batch_size = batch_size, shuffle = False,
            num_workers = workers, pin_memory = True
            )
    return valid_loader


