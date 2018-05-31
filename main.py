# This is for project Computer Vision fro Q-matrix Learning

# Official modules
import argparse
import os
import numpy as np
import torch
import time
import shutil
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Self-written modules
from models.autoencoder import AutoEncoder, DoubleAutoEncoder
from models.autoencoder import AverageMeter


parser = argparse.ArgumentParser(description='PyTorch Q-matrix Training')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# Generate Data
# np.random.seed(5)
# P_matrix = np.random.choice(a=[False, True], size=[400, 3], p=[0.5, 0.5])
# Q_matrix = np.random.choice(a=[False, True], size=[3, 9], p=[0.5, 0.5])
# R_matrix = np.matmul(P_matrix, Q_matrix).astype(float)
# np.savetxt('./data/R_matrix.csv', R_matrix, delimiter=',')

R_matrix = np.genfromtxt('./data/R_matrix.csv', delimiter=',')
R_matrix = torch.from_numpy(R_matrix)
trainData = torch.utils.data.TensorDataset(R_matrix[0:200, :])
valData = torch.utils.data.TensorDataset(R_matrix[200:300, :])
testData = torch.utils.data.TensorDataset(R_matrix[300:400, :])
best_loss = 100

R_matrix = np.genfromtxt('./data/R_matrix.csv', delimiter=',')
def main():
    global args, best_loss
    args = parser.parse_args()

    # model = AutoEncoder().cuda()
    model = AutoEncoder(items=9)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loader
    train_loader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valData, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testData, batch_size=args.batch_size, shuffle=True)

    if args.evaluate:
        test(test_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        loss1 = validate(val_loader, model, criterion)

        # remember best loss and save checkpoint
        is_best = loss1 < best_loss
        best_loss = min(loss1, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = Variable(input[0][0]).float()
        # ===================forward=====================
        input = input.reshape(1, 1, 9)
        encoded, decoded = model(input)
        # decoded = model(records)
        loss = criterion(decoded, input)
        losses.update(loss.item(), args.batch_size)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # ===================log========================
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, input in enumerate(val_loader):
            # compute output
            input = Variable(input[0][0]).float()
            input = input.reshape(1, 1, 9)
            encoded, decoded = model(input)
            loss = criterion(decoded, input)

            # record loss
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Validate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))
    return losses.avg


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # Initiate an array to store the encoded vectors of users
    arr = np.empty((0,3), float)

    # Evaluation
    with torch.no_grad():
        end = time.time()
        for i, input in enumerate(test_loader):
            # compute output
            input = Variable(input[0][0]).float()
            # input = Variable(input[0][0]).double()
            input = input.reshape(1, 1, 9)
            encoded, decoded = model(input)
            arr = np.append(arr, np.reshape(encoded.numpy(), (1, 3)), axis=0)
            loss = criterion(decoded, input)

            # record loss
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       i, len(test_loader), batch_time=batch_time, loss=losses))
                print('input:{0}'.format(input))
                print('encoded:{0}'.format(encoded))
                print('decoded:{0}'.format(decoded))
    np.savetxt('./data/estimated.csv', arr, delimiter=',')
    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()