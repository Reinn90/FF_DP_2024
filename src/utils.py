import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, random_split

from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from src import ff_mnist, ff_model, ff_cifar10

import gpustat
import time

def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt


def get_model_and_optimizer(opt, dataset):
    model = ff_model.FF_model(opt, dataset)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.linear_classifier.parameters())
    ]
    optimizer = torch.optim.SGD(
        [
            {
                "params": main_model_params,
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            },
            {
                "params": model.linear_classifier.parameters(),
                "lr": opt.training.downstream_learning_rate,
                "weight_decay": opt.training.downstream_weight_decay,
                "momentum": opt.training.momentum,
            },
        ]
    )
    return model, optimizer


def get_data(opt, partition, mnist=True):
    if mnist:
        dataset = ff_mnist.FF_MNIST(opt, partition)
    else:
        dataset = ff_cifar10.FF_CIFAR10(opt, partition)
        
    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        # num_workers=1, # changed from 4 to 1
        # persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_MNIST_partition(opt, partition):
    if partition in ["train", "val", "train_val"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition in ["test"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise NotImplementedError

    if partition == "train":
        mnist = torch.utils.data.Subset(mnist, range(50000))
    elif partition == "val":
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        mnist = torch.utils.data.Subset(mnist, range(50000, 60000))

    return mnist

def get_CIFAR10_partition(opt, partition):
    if partition in ["train", "val", "train_val"]:
        cifar10 = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition in ["test"]:
        cifar10 = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise NotImplementedError

    if partition == "train":
        cifar10 = torch.utils.data.Subset(cifar10, range(50000))
    elif partition == "val":
        cifar10 = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        cifar10 = torch.utils.data.Subset(cifar10, range(50000, 60000))
        
    # print(f"Loaded {len(cifar10)} samples for {partition} partition") #dbug
    
    return cifar10

def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t"
        f"Time: {timedelta(seconds=iteration_time)} \n",
        end="",
    )
    if scalar_outputs is not None:
        scalar_items = list(scalar_outputs.items()) #

        for i in range(0, len(scalar_items),2): #
            key1, value1 = scalar_items[i]
            if i+1 < len(scalar_items):
                key2, value2 = scalar_items[i + 1]
                print(f"{key1}: {value1:.4f}\t {key2}: {value2:.4f}")
            else:
                print(f"{key1}: {value1:.4f}")
    print()


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict


def MNIST_loaders(train_batch_size=100, test_batch_size=100, val_split=0.2):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    full_dataset = MNIST('../datasets/', train=True, download=True, transform=transform)
    
    # Split the training set into train and validation
    val_size = int(len(full_dataset) * val_split)
    
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)
    
    test_loader = DataLoader(

        MNIST("../datasets/", train=False, download=True, transform=transform),
        batch_size=test_batch_size,
        shuffle=False,
    )


    return train_loader, val_loader, test_loader


# Train network with backpropogation
def train(model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, len(data)*batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return total_loss / len(train_loader)


# Test network for backpropogation; get avg. loss
def evaluate(model, device, data_loader, loss_fn, val=False):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            total_loss += loss_fn(output, target).item() * data.size(0) # sum batch loss
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    t = 'Val' if val else 'Test'
    
    print('{} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(t,
        avg_loss, correct, len(data_loader.dataset), accuracy*100.0))
    
    return avg_loss, accuracy


def getGPUStats():
    GPUs = gpustat.new_query()
    GPU = GPUs[0]
    return GPU


def log_gpu_power(stop_event, log_queue):
    while not stop_event.is_set():
        value = getGPUStats().power_draw
        timestamp = time.time()
        log_queue.put((timestamp, value))
        #time.sleep(0.1)


def log_gpu_util(stop_event, log_queue):
    while not stop_event.is_set():
        value = getGPUStats().utilization
        timestamp = time.time()
        log_queue.put((timestamp, value))
        #time.sleep(0.1)


def log_gpu_mem(stop_event, log_queue):
    while not stop_event.is_set():
        value = getGPUStats().memory_used
        timestamp = time.time()
        log_queue.put((timestamp, value))
        #time.sleep(0.1)
