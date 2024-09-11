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

set_seed(2019)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-r', '--run_name', type=str, default="run0",
                        help="run name for tensorboard")
    parser.add_argument('-e', '--epoch', type=int, default=10,
                        help="total training epochs")
    parser.add_argument('-s', '--train_steps', type=int, default=1000,
                        help="total training steps")
    parser.add_argument('-t', '--test_steps', type=int, default=40,
                        help="steps to test")
    parser.add_argument('-v', '--save_steps', type=int, default=100,
                        help="steps to save")
    parser.add_argument('-l', '--log_steps', type=int, default=20,
                        help="steps to log")
    parser.add_argument('-w', '--warmup_steps', type=int, default=20,
                        help="warmup steps")
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_batch_norm', action="store_true")
    parser.add_argument('--use_drop_out', action="store_true")
    parser.add_argument('--drop_out_prob', type=float, default=0.5)
    parser.add_argument('--test_curve', action="store_true")
    args = parser.parse_args()
    return args

def train(model, train_loader, test_loader, 
          optimizer, scheduler, criterion,
          **train_args):
    run_name = train_args['run_name']
    epoch = train_args['epoch']
    train_steps = train_args['train_steps']
    log_step = train_args['log_steps']
    test_step = train_args['test_steps']
    save_step = train_args['save_steps']
    save_path = train_args['save_path']
    device = train_args['device']
    test_curve = train_args['test_curve']

    writer = SummaryWriter(run_name)

    model.train()
    global_step = 0
    for epoch_idx in tqdm(range(epoch), desc='train_epoch'):
        for img, label in tqdm(train_loader, desc='train_batch'):
            if global_step > train_steps:
                return

            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            logits = model(img)
            loss = criterion(logits, label)
            loss.backward()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Learning Rate', current_lr, global_step)

            if global_step % log_step == 0:
                # log to tensorboard here
                writer.add_scalar('Loss/train_loss', loss.item(), global_step)
                print('train loss:', loss.item())
                label_pred = torch.argmax(logits, -1)
                acc = torch.sum(label == label_pred).float() / label_pred.shape[0]
                writer.add_scalar('Metrics/train_acc', acc.item(), global_step)
                print('train acc:', acc.item())

            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % test_step == 0:
                eval(model, 
                     train_loader, 
                     test_loader, 
                     criterion, 
                     writer, 
                     device, 
                     global_step, 
                     test_curve)

            if global_step % save_step == 0:
                torch.save(model, os.path.join(save_path, f'model_{global_step}'))

@torch.no_grad()
def eval(model, train_loader, test_loader, criterion, writer, device, global_step, test_curve):
    model.eval()

    if test_curve:
        train_acc_total = []
        for img, label in tqdm(train_loader, desc='eval on train batch'):
            img = img.to(device)
            label = label.to(device)
            logits = model(img)

            label_pred = torch.argmax(logits, -1)
            acc = torch.sum(label == label_pred).float() / label_pred.shape[0]
            train_acc_total.append(acc.item())
        acc = np.mean(np.array(train_acc_total))
        writer.add_scalar('Metrics/train_accuracy_full', acc, global_step)
        print('train_accuracy_full:', acc)

    val_loss_total = []
    acc_total = []
    for img, label in tqdm(test_loader, desc='val_batch'):
        img = img.to(device)
        label = label.to(device)
        logits = model(img)

        val_loss = criterion(logits, label)
        val_loss_total.append(val_loss.item())

        label_pred = torch.argmax(logits, -1)
        acc = torch.sum(label == label_pred).float() / label_pred.shape[0]
        acc_total.append(acc.item())

    val_loss = np.mean(np.array(val_loss_total))
    acc = np.mean(np.array(acc_total))
    if not test_curve:
        writer.add_scalar('Loss/val_loss', val_loss, global_step)
        writer.add_scalar('Metrics/val_accuracy', acc, global_step)
        print('val_loss:', val_loss)
        print('val_acc:', acc)
    else:
        writer.add_scalar('Loss/test_loss', val_loss, global_step)
        writer.add_scalar('Metrics/test_accuracy', acc, global_step)
        print('test_loss:', val_loss)
        print('test_acc:', acc)

    

    model.train()

def lr_cos(current_step: int, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))))
    )


def main():
    args = parse_arg()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # load dataset
    fashion_mnist = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)

    bs = args.batch_size
    if not args.test_curve:
        test_ratio = 0.2
        test_size = int(test_ratio * len(fashion_mnist))
        train_size = len(fashion_mnist) - test_size
        train_dataset, test_dataset = random_split(fashion_mnist, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    else:
        train_loader = DataLoader(fashion_mnist, batch_size=bs, shuffle=True)
        fashion_mnist_test = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)
        test_loader = DataLoader(fashion_mnist_test, batch_size=bs, shuffle=True)

    # set lr
    if args.use_batch_norm:
        model = LeNet5(num_class=10, use_batch_norm=True)
    elif args.use_drop_out:
        model = LeNet5(num_class=10, use_drop_out=True, drop_out_prob=args.drop_out_prob)
    else:
        model = LeNet5(num_class=10)
    train_steps = args.train_steps
    warmup_steps = args.warmup_steps
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_cos(step, warmup_steps, train_steps))
    criterion = nn.CrossEntropyLoss()

    # check that GPU is available
    if torch.cuda.is_available():
        gpu = args.gpu
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    # set training args
    save_dir = args.save_dir
    save_path = os.path.join(save_dir, args.run_name)
    os.makedirs(save_path, exist_ok=True)

    train_args = {
        'run_name': args.run_name,
        'epoch': args.epoch,
        'train_steps': args.train_steps,
        'test_steps': args.test_steps,
        'save_steps': args.save_steps,
        'log_steps': args.log_steps,
        'save_path': save_path,
        'device': device,
        'test_curve': args.test_curve,
    }

    # train model
    model = model.to(device)
    train(model=model, 
          train_loader=train_loader, 
          test_loader=test_loader, 
          optimizer=optimizer, 
          scheduler=scheduler,
          criterion=criterion,
          **train_args)

if __name__ == '__main__':
    main()