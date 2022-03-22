import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import numpy as np
from IPython import display
import requests
from io import BytesIO
from PIL import Image
from PIL import Image, ImageSequence
from IPython.display import HTML
import warnings
from matplotlib import rc
import gc
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
gc.enable()
plt.ioff()


def set_model():
    num_classes = 10
    resnet = resnet18(pretrained=True)
    resnet.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1)
    resnet_ = list(resnet.children())[:-2]
    resnet_[3] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    classifier = nn.Conv2d(512,num_classes,1)
    torch.nn.init.kaiming_normal_(classifier.weight)
    resnet_.append(classifier)
    resnet_.append(nn.Upsample(size=32, mode='bilinear', align_corners=False))
    tiny_resnet = nn.Sequential(*resnet_)
    return tiny_resnet

def set_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(root='/root/dataset/CIFAR', train=True, download=True, transform=transform_train)
    train_iter = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    testset = CIFAR10(root='/root/dataset/CIFAR', train=False, download=True, transform=transform_test)
    test_iter = DataLoader(testset, batch_size=100, shuffle=False, num_workers=16, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_iter, test_iter, classes


def attention(x):
    return torch.sigmoid(torch.logsumexp(x,1, keepdim=True))

def main():
    trainloader, testloader, class_name = set_data()
    model = nn.DataParallel(set_model()).cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,78,eta_min=0.001)

    num_epochs = 50
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        acc = 0.0
        var = 0.0
        model.train()
        train_pbar = trainloader
        for i, (x, _label) in enumerate(train_pbar):
            x = x.cuda()
            _label = _label.cuda()
            label = F.one_hot(_label).float()
            seg_out = model(x)
            
            attn = attention(seg_out)
            # Smooth Max Aggregation
            logit = torch.log(torch.exp(seg_out*0.5).mean((-2,-1)))*2
            loss = criterion(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            acc += (logit.argmax(-1)==_label).sum()
    return 0

if __name__ == "__main__":
    main()