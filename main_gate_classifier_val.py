# latest code, uses 45/5k validation set
# final code for cifar training
# this code logs gate value g during training
# for now, we perform experiments on only HP: scratch init, it1g-2cm-1
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import *
import csv
import pandas as pd

import argparse
import os
import time
from utils import progress_bar
from torch.utils.tensorboard import SummaryWriter
import models.resnet_gate_classifier as models
import itertools
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar 10 Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet_gate_classifier',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=110, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck')
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=333, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[133, 200],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='baseline_models', type=str)
parser.add_argument('--lr_decay_step', default=50, type=int, help='step size after which learning rate is decayed')
parser.add_argument('--lr_decay_gamma', default=0.1, type=float,
                    help='gamma value for lr decay')
parser.add_argument('--test_checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--s_no', default=1, type=int,
                    help='to track which out of the 5 trained models of the same model it is')
parser.add_argument('--lambda_mem', default=0.5, type=float,
                    help='memorization loss hyperparameter')
parser.add_argument('--scratch', default=1, type=int,
                    help='train from scratch or use the baseline to initialise part of te model')
parser.add_argument('--gate_iters', default=1, type=int,
                    help='the number of times the gate classifer is to be trained')
parser.add_argument('--lr_gate', '--learning-rate_gate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate for the gate FC block')
parser.add_argument('--schedule_gate', default=0, type=int,
                    help='0 if want to keep gate lr constant, 1 for reducing it by 0.1 every 50 epochs')
parser.add_argument('--mod_name_suffix', default='', type=str,
                    help='name to make model easily identifiable')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Test if cuda is working or not: ", device)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
np.random.seed(0)

if args.scratch==1:
    p = 'scratch'
else:
    p = 'bi'
print('check initialisation:   ', p)

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

def main():

    best_acc = 0  # best validation set accuracy
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..%s' % args.dataset)
    tf_dir = 'vc{}_resgc{}_4770_{}_{}'.format(args.dataset[5:], str(args.depth), p, args.mod_name_suffix) #resgc - resnet_gate_classifier
    writer = SummaryWriter('cifar_resnets_modified_val/' + tf_dir) #***
    transform_train = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),])
    transform_test = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),])

    if args.dataset == 'cifar10':
        dataloader = torchvision.datasets.CIFAR10
        num_classes = 10
        baseline_dir = './baselines/c10_resnet110_3248/checkpoint_final.pth' #***
    else:
        dataloader = torchvision.datasets.CIFAR100
        num_classes = 100
        baseline_dir = './baselines/c100_resnet164_3248/checkpoint_final.pth' #***
    data_dir = './data' #***
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    trainset = dataloader(root=data_dir, train=True, download=True, transform=transform_train)
    valset = dataloader(root=data_dir, train=True, download=True, transform=transform_test)
    testset = dataloader(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    split = int(np.floor(0.1 * num_train)) #using a 45k/5k train val split, i.e. its a 10% split
    valid_idx, train_idx = indices[:split], indices[split:]

    print('\nMeta data, ')
    print('\ntensorboard logs directory', tf_dir)
    print('\nargs.scratch {}, args.gate_iters {}, args.lr_gate {}, args.schedule_gate {}, args.schedule {}, args.epochs {}'.format(args.scratch, args.gate_iters, args.lr_gate, args.schedule_gate, args.schedule, args.epochs))
    print('\ntrain size {}, val size {}'.format(len(train_idx), len(valid_idx)))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers) ###to check again
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.workers)

    if args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes = num_classes,
                    depth = args.depth,
                    block_name = args.block_name,
                    )

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # defining different optimizers for different losses backprop
    list_parameter = [
    model.module.conv1.parameters(),
    model.module.bn1.parameters(),
    model.module.layer1.parameters(),
    model.module.layer2.parameters(),
    model.module.o1.parameters(),
    model.module.layer3.parameters(),
    model.module.o2.parameters(),
    model.module.fc.parameters()]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(itertools.chain(*list_parameter), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_gate = torch.optim.SGD(model.module.g_layer.parameters(), args.lr_gate,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            new_dict_load = {}
            for k, v in checkpoint['state_dict'].items():
                new_dict_load['module.'+k] = v
            model.load_state_dict(new_dict_load)
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            args.lr_gate = 0.01
            for param_group in optimizer_gate.param_groups:
                param_group['lr'] = args.lr_gate
            best_acc = checkpoint['acc']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format('yes', checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # to be modified specific to the layers of the resnet being used
    # if args.scratch is false, then load the baseline model
    if args.scratch == 0 and not args.resume:
        if args.block_name.lower() == 'basicblock':
            n = (args.depth - 2) // 6
        elif args.block_name.lower() == 'bottleneck':
            n = (args.depth - 2) // 9

        ckpt = torch.load(baseline_dir)
        print("loaded checkpoint\n", baseline_dir)
        new_dict_to_load = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.layer1') or k.startswith('module.layer2') or k.startswith('module.conv') or k.startswith('module.bn'):
                new_dict_to_load[k] = v
            elif k.startswith('module.layer3.0.'):
                new_key = k.replace('module.layer3.0.', 'module.o1.')
                new_dict_to_load[new_key] = v
            elif k.startswith('module.layer3.{}.'.format(n-1)):
                new_key = k.replace('module.layer3.{}.'.format(n-1), 'module.o2.')
                new_dict_to_load[new_key] = v
            elif k.startswith('module.layer3.'):
                int_ = int(k[14])
                new_key = k
                new_key[14] = int_ - 1
                new_dict_to_load[new_key] = v

        model.load_state_dict(new_dict_to_load, strict=False)

    if args.evaluate:
        print("Evaluation only\n")
        path = os.path.join('path to checkpoint directory', args.test_checkpoint)
        if os.path.isfile(path):
            print("=> loading test checkpoint")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
        acc, test_loss = validate(testloader, model, criterion)
        print("Test accuracy attained: {}, Test loss: {} ".format(acc, test_loss))
        return

    if args.depth in [1202, 110, 164] and not args.resume:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this implementation it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01


    test_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, optimizer_gate, epoch)
        if epoch == 1: # check for 2nd epoch
            # after the 1st epoch (0th), bring it back to initial lr = 0.1 and let the normal lr schedule follow
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        print('\nEpoch: [%d | %d] LR: %f  LR_gate: %f' % (epoch, args.epochs - 1, optimizer.param_groups[0]['lr'], optimizer_gate.param_groups[0]['lr']))

        # train for one epoch
        train(trainloader, model, criterion, optimizer, optimizer_gate, epoch)

        # evaluate on validation set
        val_acc, val_loss, val_loss_cls, val_loss_gatecel, g_hist_val  = validate(validloader, model, criterion, 1)
        test_acc, test_loss, test_loss_cls, test_loss_gatecel, g_hist_test  = validate(testloader, model, criterion, 1)
        train_acc, train_loss, train_loss_cls, train_loss_gatecel, g_hist_train = validate(trainloader, model, criterion, 1)

        info = {'val_loss_cls': val_loss_cls, 'val_loss_gatecel': val_loss_gatecel,  'tot_val_loss': val_loss, 'val_acc': val_acc,
                'train_loss_cls': train_loss_cls, 'train_loss_gatecel': train_loss_gatecel, 'tot_train_loss': train_loss, 'train_acc': train_acc,
                'tot_test_loss': test_loss, 'test_acc': test_acc, 'lr': optimizer.param_groups[0]['lr'], 'gate_lr': optimizer_gate.param_groups[0]['lr']}

        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch+1)
        writer.add_histogram('g Values, train', np.array(g_hist_train), global_step=epoch+1)
        writer.add_histogram('g Values, validation', np.array(g_hist_val), global_step=epoch+1)

        # remember best prec@1 and save checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'acc': best_acc,
                'optimizer': optimizer.state_dict(),
                }, 'checkpoint_best.pth')

        if epoch % 50 == 0 and epoch!=0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'acc': test_acc,
                'optimizer': optimizer.state_dict(),
                }, 'checkpoint_{}.pth'.format(epoch))

    # saving g values on test set to .csv file
    df = pd.DataFrame({"g value" : np.array(g_hist_test)})
    name = 'g_value' + tf_dir + '.csv'
    df.to_csv(name, index=True)
    print("written g testset")
    print("final test accuracy at the end of {} epochs: ".format(args.epochs), test_acc)
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'acc': test_acc,
                'optimizer': optimizer.state_dict(),
                }, 'checkpoint_final_{}.pth'.format(epoch+1))



def train(trainloader, model, criterion, optimizer, optimizer_gate, poch):
    """
        Run one train epoch
    """
    model.train()
    correct = 0
    total = 0
    tot_loss = 0
    flag = 0
    g_vector_hist = []
    for ind, (inputs, target) in enumerate(trainloader):
        inputs = inputs.to(device)
        target = target.to(device)

        # compute the gate classifier loss and backprop it through only the gate FC block defined by g_layer
        model_output, gate_block_out, _ = model(inputs)
        gate_cel = criterion(gate_block_out, target)

        optimizer_gate.zero_grad()
        gate_cel.backward(retain_graph=True)
        optimizer_gate.step()

        # backprop classification loss only when ind%args.gate_iters == 0
        if (ind+1) % args.gate_iters == 0:
            # calculate the main classification loss
            classification_loss = criterion(model_output, target)

            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()

            tot_loss += classification_loss.item()

        _, predicted = model_output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        progress_bar(ind, len(trainloader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
            % (tot_loss/(ind+1), 100.*correct/total, correct, total))


def validate(testloader, model, criterion, flag=0):
    """
    Run evaluation
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_loss_gatecel = 0.0
    test_loss_cls = 0.0

    with torch.no_grad():
        g_vector_hist = []
        for i, (inputs, target) in enumerate(testloader):
            inputs, target = inputs.to(device), target.to(device)
            output, gate_out, g_value = model(inputs)

            loss_gate = criterion(gate_out, target)
            classification_loss = criterion(output, target)
            loss = loss_gate + classification_loss

            test_loss += loss.item()
            test_loss_cls += classification_loss.item()
            test_loss_gatecel += loss_gate.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            #if not flag:
            #    progress_bar(i, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #        % (test_loss/(i+1), 100.*correct/total, correct, total))
            if flag:
                g_vector_hist.extend(g_value.cpu())

    acc = 100.*correct/total
    return acc, test_loss/total, test_loss_cls/total, test_loss_gatecel/total, g_vector_hist


def adjust_learning_rate(optimizer, optimizer_gate, epoch):
    """Sets the learning rate to the present LR decayed by lr_decay_gamma at every point in the schedule"""
    if epoch == args.schedule[0]:
        lr = args.lr*(args.lr_decay_gamma**1)
    elif epoch == args.schedule[1]:
        lr = args.lr*(args.lr_decay_gamma**2)
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if args.schedule_gate!=0 and (epoch+1) % 50 == 0:
        lr_gate = args.lr_gate * (0.1 ** ((epoch+1) // 50))
        for param_group in optimizer_gate.param_groups:
            param_group['lr'] = lr_gate


def save_checkpoint(state, name):
    """
    Save the training model
    """
    #***create the directory '/checkpoint/scratch' and ''/checkpoint/bi'
    path = './checkpoint/' + p + '/vc{}_resgc{}_4770_{}'.format(args.dataset[5:], str(args.depth), args.mod_name_suffix)
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(state, os.path.join(path, name))
    print("checkpoint saved: ", name)


if __name__ == '__main__':
    main()
