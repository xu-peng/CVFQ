import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, items=9):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(items, 3),
            # nn.ReLU(True))
            nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Linear(3, items),
            nn.Sigmoid())

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec


class DoubleAutoEncoder(nn.Module):
    def __init__(self, items=9):
        super(DoubleAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=0, bias=False),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            # nn.Linear(items, 3),
            # nn.ReLU(True))
            nn.Conv1d(256, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Linear(3, items),
            nn.Sigmoid())

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec



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