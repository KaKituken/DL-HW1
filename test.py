import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from model import LeNet5

from tqdm import tqdm
import math
import random

import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, default='./save')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2019)
    args = parser.parse_args()
    return args


@torch.no_grad()
def eval(model, test_loader, device):
    model.eval()

    acc_total = []
    for img, label in tqdm(test_loader, desc='val_batch'):
        img = img.to(device)
        label = label.to(device)
        logits = model(img)

        label_pred = torch.argmax(logits, -1)
        acc = torch.sum(label == label_pred).float() / label_pred.shape[0]
        acc_total.append(acc.item())

    acc = np.mean(np.array(acc_total))
    print('val_acc:', acc)



def main():
    args = parse_arg()
    seed = args.seed
    set_seed(seed)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # load dataset
    dataset = args.dataset

    bs = args.batch_size
    fashion_mnist_test = datasets.FashionMNIST(dataset, download=True, train=False, transform=transform)
    test_loader = DataLoader(fashion_mnist_test, batch_size=bs, shuffle=True)


    ckpt = args.checkpoint
    # load checkpoint
    model = torch.load(ckpt)

    # check that GPU is available
    if torch.cuda.is_available():
        gpu = args.gpu
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    # set training args
    model = model.to(device)
    eval(model=model, 
         test_loader=test_loader, 
         device=device)

if __name__ == '__main__':
    main()