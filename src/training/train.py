#!/usr/bin/env python3
"""
Ahmed Taibeche
water bottle visual inspection based on deep learning project
using pytorch
TRAIN THE MODELS

usage: python -m src.training.train data/bottles --arch inception_v3 --model-dir models/inception_v3

saves model_best.pth.tar (best validation accuracy), checkpoint.pth.tar (last epoch),
labels.txt (class names) and the tensorboard logs to --model-dir
"""
import argparse
import os
import random

import time
import shutil
import warnings
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from src.training.reshape import SUPPORTED_ARCHS, create_model

# get the available network architectures that can be reshaped for transfer learning
model_names = sorted(name for name in models.list_models(module=models)
    if name.startswith(SUPPORTED_ARCHS))


# parse command-line arguments
parser = argparse.ArgumentParser(description='PyTorch Image Classifier Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset (train/ and val/ folders with one subfolder per class)')
parser.add_argument('--model-dir', type=str, default='models',
                    help='path to desired output directory for saving model '
                         'checkpoints (default: models/)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture, e.g. resnet18, resnet50, inception_v3, '
                         'mobilenet_v3_large, efficientnet_b0 (default: resnet18)')
parser.add_argument('--resolution', default=None, type=int, metavar='N',
                    help='input NxN image resolution of model '
                         '(default: 299 for Inception models, 224 otherwise)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 50)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                    help='mini-batch size (default: 8)')
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'],
                    help='optimizer (default: adam)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-4)', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of the SGD optimizer (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--dropout', default=0.0, type=float, metavar='P',
                    help='add a dropout layer with probability P before the new '
                         'classification layer (default: 0, disabled)')
parser.add_argument('--freeze-backbone', action='store_true',
                    help='only train the new classification layers and keep the '
                         'pre-trained layers frozen')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                    help='train from randomly initialized weights instead of the ImageNet weights')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU ID to use when CUDA is available (default: 0)')

# variable for storing the best model accuracy so far
best_accuracy = 0


def main(args):
    """
    Load dataset, setup model, and train for N epochs
    """
    global best_accuracy
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # train on the GPU when available, on the CPU otherwise
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"=> using GPU {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        device = torch.device("cpu")
        print("=> CUDA is not available, training on the CPU")

    # setup data transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    train_dataset = datasets.ImageFolder(os.path.join(args.data, 'train'), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'), val_transforms)

    print(f"=> dataset classes:  {len(train_dataset.classes)}  {train_dataset.classes}")

    # save the class names next to the checkpoints
    save_labels(train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == 'cuda')

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == 'cuda')

    # create the model (pre-trained on ImageNet by default) and reshape it for the number of classes in the dataset
    model = create_model(args.arch, len(train_dataset.classes), args.pretrained,
                         args.dropout, args.freeze_backbone)
    model = model.to(device)

    # define loss function (criterion) and optimizer, for the trainable parameters only
    criterion = nn.CrossEntropyLoss().to(device)
    parameters = [param for param in model.parameters() if param.requires_grad]

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(parameters, args.lr,
                                     weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            best_accuracy = checkpoint['best_accuracy']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    cudnn.benchmark = True

    # if in evaluation mode, only run validation
    if args.evaluate:
        val_loss, val_acc = validate(val_loader, model, criterion, 0, device)
        print(f"  * Val Loss       {val_loss:.4e}")
        print(f"  * Val Accuracy   {val_acc:.4f}")
        return

    # train for the specified number of epochs
    for epoch in range(args.start_epoch, args.epochs):
        # decay the learning rate
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device)

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, device)

        # remember best acc@1 and save checkpoint
        is_best = val_acc > best_accuracy
        best_accuracy = max(val_acc, best_accuracy)

        print(f"=> Epoch {epoch}")
        print(f"  * Train Loss     {train_loss:.4e}")
        print(f"  * Train Accuracy {train_acc:.4f}")
        print(f"  * Val Loss       {val_loss:.4e}")
        print(f"  * Val Accuracy   {val_acc:.4f}{'*' if is_best else ''}")

        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'resolution': args.resolution,
            'classes': train_dataset.classes,
            'num_classes': len(train_dataset.classes),
            'pretrained': args.pretrained,
            'dropout': args.dropout,
            'state_dict': model.state_dict(),
            'accuracy': {'train': train_acc, 'val': val_acc},
            'loss' : {'train': train_loss, 'val': val_loss},
            'optimizer' : optimizer.state_dict(),
            'best_accuracy': best_accuracy,
        }, is_best)


def compute_loss(output, target, criterion):
    """
    Compute the loss of a batch, including the auxiliary classifiers that
    Inception and GoogleNet return in training mode. Returns (logits, loss)
    """
    if isinstance(output, models.InceptionOutputs):
        loss = criterion(output.logits, target) + 0.4 * criterion(output.aux_logits, target)
        return output.logits, loss

    if isinstance(output, models.GoogLeNetOutputs):
        aux_loss = criterion(output.aux_logits1, target) + criterion(output.aux_logits2, target)
        return output.logits, criterion(output.logits, target) + 0.3 * aux_loss

    return output, criterion(output, target)


def train(train_loader, model, criterion, optimizer, epoch, device):
    """
    Train one epoch over the dataset
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Accuracy', ':7.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc],
        prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    model.train()

    # get the start time
    epoch_start = time.time()
    end = epoch_start

    # train over each image batch from the dataset
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output, loss = compute_loss(model(images), target, criterion)

        # record loss and measure accuracy
        losses.update(loss.item(), images.size(0))
        acc.update(accuracy(output, target), images.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(train_loader)-1:
            progress.display(i)

    print(f"Epoch: [{epoch}] completed, elapsed time {time.time() - epoch_start:6.3f} seconds")

    tensorboard.add_scalar('Loss/train', losses.avg, epoch)
    tensorboard.add_scalar('Accuracy/train', acc.avg, epoch)

    return losses.avg, acc.avg


def validate(val_loader, model, criterion, epoch, device):
    """
    Measure model performance across the val dataset
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Accuracy', ':7.3f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc],
        prefix='Val:   ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output, loss = compute_loss(model(images), target, criterion)

            # record loss and measure accuracy
            losses.update(loss.item(), images.size(0))
            acc.update(accuracy(output, target), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(val_loader)-1:
                progress.display(i)

    tensorboard.add_scalar('Loss/val', losses.avg, epoch)
    tensorboard.add_scalar('Accuracy/val', acc.avg, epoch)

    return losses.avg, acc.avg


def save_labels(classes, labels_filename='labels.txt'):
    """
    Save the class names (one per line, in model output order) to the model directory
    """
    model_dir = os.path.expanduser(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    labels_filename = os.path.join(model_dir, labels_filename)

    with open(labels_filename, 'w') as file:
        for label in classes:
            file.write(f"{label}\n")

    print(f"saved class labels to:  {labels_filename}")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    """
    Save a model checkpoint file, along with the best-performing model if applicable
    """
    if args.model_dir:
        model_dir = os.path.expanduser(args.model_dir)
        os.makedirs(model_dir, exist_ok=True)

        filename = os.path.join(model_dir, filename)
        best_filename = os.path.join(model_dir, best_filename)

    # save the checkpoint
    torch.save(state, filename)

    # earmark the best checkpoint
    if is_best:
        shutil.copyfile(filename, best_filename)
        print(f"saved best model to:  {best_filename}")
    else:
        print(f"saved checkpoint to:  {filename}")


def adjust_learning_rate(optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """
    Computes the accuracy of predictions vs groundtruth
    """
    with torch.no_grad():
        preds = (output.argmax(dim=-1) == target)
        return preds.float().mean().cpu().item() * 100.0


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Progress metering
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    args = parser.parse_args()

    # Inception models are pre-trained at 299x299, the other models at 224x224
    if args.resolution is None:
        args.resolution = 299 if args.arch.startswith('inception') else 224

    # open tensorboard logger (to model_dir/tensorboard)
    tensorboard = SummaryWriter(log_dir=os.path.join(args.model_dir, "tensorboard", f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    print(f"To start tensorboard run:  tensorboard --logdir={os.path.join(args.model_dir, 'tensorboard')}")

    main(args)
