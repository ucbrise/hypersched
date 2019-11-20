import math
import os
import time
import sys
import torch
import torchvision

# if __name__ == '__main__':
#     torch.multiprocessing.set_start_method('spawn')

import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.multiprocessing import Pool, Process

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import ray
import sys
import numpy as np

from ..utils import TimerStat

TRANSFORMS = {
    "cifar": [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
    "imagenet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
}

DATASET = {
    "cifar": torchvision.datasets.CIFAR10,
    "imagenet": torchvision.datasets.ImageNet,
}

CROP = {"cifar": 32, "imagenet": 224}


def get_cifar_dataset():
    return mass_download("cifar")


def mass_download(dataset, download=True):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(CROP[dataset], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*TRANSFORMS[dataset]),
        ]
    )  # meanstd transformation

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*TRANSFORMS[dataset]),]
    )
    from filelock import FileLock

    config = {
        "root": os.path.expanduser("~/data_cifar"),
        "train": True,
        "download": download,
        "transform": transform_train,
    }
    with FileLock(os.path.expanduser(f"~/data_{dataset}.lock")):
        trainset = DATASET[dataset](**config)
    config["train"] = False
    valset = DATASET[dataset](**config)
    return trainset, valset


def imagenet_dataset():
    trainset = torchvision.datasets.ImageNet(
        root="/data/imagenet",
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        ),
        split="train",
    )
    valset = torchvision.datasets.ImageNet(
        root="/data/imagenet",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        ),
        split="val",
    )
    return trainset, valset


def prefetch(dataset, resource="GPU"):
    devices_in_cluster = int(ray.cluster_resources().get(resource))
    remote_download = ray.remote(num_gpus=1)(mass_download)
    ray.get(
        [
            remote_download.remote(dataset.lower())
            for i in range(devices_in_cluster)
        ]
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


# def train(train_loader, model, criterion, optimizer, max_steps=None):
def train(
    train_loader, model, criterion, optimizer, epoch, max_steps=None
):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    timers = {k: TimerStat() for k in ["d2h", "fwd", "grad", "apply"]}

    # switch to train mode
    model.train()

    end = time.time()
    train_start = time.time()
    for idx, (features, target) in enumerate(train_loader):
        if max_steps and idx > max_steps:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        # Create non_blocking tensors for distributed training
        with timers["d2h"]:
            features = features.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        with timers["fwd"]:
            output = model(features)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), features.size(0))
            top1.update(prec1[0], features.size(0))

        with timers["grad"]:
            # compute gradients in a backward pass
            optimizer.zero_grad()
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        with timers["apply"]:
            # Call step of optimizer to update model params
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    stats = {
        "train_accuracy": top1.avg.cpu(),
        "batch_time": batch_time.avg,
        "train_loss": losses.avg,
        "data_time": data_time.avg,
        "train_time": time.time() - train_start,
    }
    stats.update({k: t.mean for k, t in timers.items()})
    return stats


def state_to_cuda(state):
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()
    return state


def state_from_cuda(state):
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cpu()
    return state


class Adjuster:
    def __init__(self, initial, target=None, steps=None, mode=None):
        self.mode = mode
        self._initial = initial
        self._lr = initial
        self._target = initial
        if target and self.mode is not None:
            self._target = max(target, initial)
        self.steps = steps or 10
        print(f"Creating an adjuster from {initial} to {target}: mode {mode}")

    def adjust(self):
        if self._lr < self._target:
            diff = (self._target - self._initial) / self.steps
            self._lr += min(diff, 0.1 * self._initial)
        return self._lr

    @property
    def current_lr(self):
        return self._lr


def adjust_learning_rate(initial_lr, optimizer, epoch, decay=True):
    optim_factor = 0
    if decay:
        if epoch > 160:
            optim_factor = 3
        elif epoch > 120:
            optim_factor = 2
        elif epoch > 60:
            optim_factor = 1

    lr = initial_lr * math.pow(0.2, optim_factor)
    # lr = initial_lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


######################################################################
# Validation Function
# ~~~~~~~~~~~~~~~~~~~
#
# To track generalization performance and simplify the main loop further
# we can also extract the validation step into a function called
# ``validate``. This function runs a full validation step of the features
# model on the input validation dataloader and returns the top-1 accuracy
# of the model on the validation set. Again, you will notice the only
# distributed training feature here is setting ``non_blocking=True`` for
# the training data and labels before they are passed to the model.
#


def validate(val_loader, model, criterion, max_steps=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    correct = 0
    total = 0
    val_start = time.time()
    with torch.no_grad():
        end = time.time()
        for idx, (features, target) in enumerate(val_loader):
            if max_steps and idx > max_steps:
                break

            features = features.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(features)
            loss = criterion(output, target)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), features.size(0))
            # top1.update(prec1[0], features.size(0))
            # top5.update(prec5[0], features.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % 100 == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
            #            i, len(val_loader), batch_time=batch_time, loss=losses,
            #            top1=top1))
    # acc = #top1.avg if type(top1.avg) == int else top1.avg.cpu()
    acc = correct / total
    stats = {
        "batch_time": batch_time.avg,
        "mean_accuracy": acc,  # for backward compat
        "val_accuracy": acc,
        "neg_mean_accuracy": -acc,
        "mean_loss": losses.avg,
        "val_loss": losses.avg,  # for backward compat
        "val_time": time.time() - val_start,
    }
    return stats
