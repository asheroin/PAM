#!/usr/bin/env python
# coding=utf-8


import torch

import time

print_freq = 10

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count= 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # set model mode
    model.train()
    end = time.time()

    for i, iter_item in enumerate(train_loader):
        task_type = None
        if len(iter_item)==2:
            input, target = iter_item
            task_type = 'Single'
        elif len(iter_item)==3:
            input, mask, target = iter_item
            mask = torch.tensor(mask)
            target = torch.tensor(target)
            task_type = 'Multi'
            mask = mask.cuda(async = True)
            mask_var = torch.autograd.Variable(mask)
        else:
            raise Exception, 'Unknow input'
        # measure data loading time
        data_time.update(time.time() - end)
        # convert to cuda
        target = target.cuda(async = True)
        # set autograd
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        if task_type == 'Multi':
            loss = (mask_var.float() * criterion(output, target_var.float())).mean()
        else:
            loss = criterion(output, target_var.float().view(-1,1)).mean()
        # measure accuracy and record loss
        # prec1, prec5 = accurary(output.data, target, topk=(1,5))
        losses.update(loss.data.item(), input.size(0))
        # top1.update(prec1.item(), input.size(0))
        # top5.update(prec5.item(), input.size(0))
        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print information
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time = batch_time,
                        data_time = data_time, loss = losses
                        ))
    return None

def valid(val_loader, model, criterion,epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    # set model mode
    model.eval()
    end = time.time()
    for i, iter_item in enumerate(val_loader):
        task_type = None
        if len(iter_item)==2:
            input, target = iter_item
            task_type = 'Single'
        elif len(iter_item)==3:
            input, mask, target = iter_item
            mask = torch.tensor(mask)
            target = torch.tensor(target)
            task_type = 'Multi'
            mask = mask.cuda(async = True)
            mask_var = torch.autograd.Variable(mask)
        else:
            raise Exception, 'Unknow input'
        # measure data loading time
        data_time.update(time.time() - end)
        # convert to cuda
        target = target.cuda(async = True)
        # set autograd
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        if task_type == 'Multi':
            loss_no_mask = criterion(output, target_var.float())
            loss = (mask_var.float() * loss_no_mask).mean()
        else:
            loss = criterion(output, target_var.float().view(-1,1)).mean()
        # measure accuracy and record loss
        losses.update(loss.data.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print information
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(val_loader), batch_time = batch_time,
                        data_time = data_time, loss = losses
                        ))
    return losses.avg

def evaluate(eval_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    # set model mode
    model.eval()
    end = time.time()
    target_list = []
    output_list = []
    for i, iter_item in enumerate(eval_loader):
        task_type = None
        if len(iter_item)==2:
            input, target = iter_item
            task_type = 'Single'
        elif len(iter_item)==3:
            input, mask, target = iter_item
            mask = torch.tensor(mask)
            target = torch.tensor(target)
            task_type = 'Multi'
            mask = mask.cuda(async = True)
            mask_var = torch.autograd.Variable(mask)
        else:
            raise Exception, 'Unknow input'
        if task_type == 'Multi':
            batch_num,_ = target.shape
            target_list.extend([target[x, :].cpu().numpy() for x in range(batch_num)])
        else:
            batch_num = target.shape[0]
            target_list.extend([x for x in target.cpu()])
        # measure data loading time
        data_time.update(time.time() - end)
        # convert to cuda
        target = target.cuda(async = True)
        # set autograd
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        # multi output fixs
        output_data_numpy = output.cpu().detach().numpy()
        if task_type == 'Multi':
            batch_num, _ = output_data_numpy.shape
            output_list.extend([list(output_data_numpy[x, :]) for x in range(batch_num)])
        else:
            output_list.extend([x[0] for x in output_data_numpy])
        if task_type == 'Multi':
            loss_no_mask = criterion(output, target_var.float())
            loss = (mask_var.float() * loss_no_mask).mean()
        else:
            loss = criterion(output, target_var.float().view(-1,1)).mean()
        data_time = AverageMeter()
        # measure accuracy and record loss
        losses.update(loss.data.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print information
        if i % print_freq == 0:
            print('Batch: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        i, len(eval_loader), batch_time = batch_time,
                        data_time = data_time, loss = losses
                        ))
    print('target_list lenght: {}'.format(len(target_list)))
    print('output_list lenght: {}'.format(len(output_list)))
    return losses.avg, target_list, output_list


def train_multi_task(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # set model mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # convert to cuda
        target = target.cuda(async = True)
        # set autograd
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var.float().view(-1,1))
        # measure accuracy and record loss
        # prec1, prec5 = accurary(output.data, target, topk=(1,5))
        losses.update(loss.data.item(), input.size(0))
        # top1.update(prec1.item(), input.size(0))
        # top5.update(prec5.item(), input.size(0))
        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print information
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time = batch_time,
                        data_time = data_time, loss = losses
                        ))
    return None
