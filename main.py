import time
from collections import defaultdict

import hydra
import torch
from tqdm import tqdm
from omegaconf import DictConfig

from src import utils


def train(opt, model, optimizer, mnist=True):

    print("MNIST") if mnist else print("CIFAR10")
    
    start_time = time.time()
    train_loader = utils.get_data(opt, "train", mnist)
    num_steps_per_epoch = len(train_loader)

    for epoch in tqdm(range(opt.training.epochs)):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()

            optimizer.step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate_or_test(opt, model, "val", mnist, epoch=epoch)

    return model


def validate_or_test(opt, model, partition, mnist=True, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, partition, mnist)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)
    model.train()


@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    # True = mnist, False = cifar10
    mnist_T_cifar_F = True
    
    model_start_time = time.time()
    opt = utils.parse_args(opt)
    model, optimizer = utils.get_model_and_optimizer(opt, "mnist") # mnist/cifar10
    model = train(opt, model, optimizer, mnist_T_cifar_F) 
    validate_or_test(opt, model, "val", mnist_T_cifar_F)  

    if opt.training.final_test:
        validate_or_test(opt, model, "test", mnist_T_cifar_F) 

    model_total_time = time.time() - model_start_time
    print(f"FF training time: {model_total_time//60}min {model_total_time%60:.2f}sec")


if __name__ == "__main__":
    my_main()
