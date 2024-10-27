import time
from collections import defaultdict
import queue
import threading
import pandas as pd
import matplotlib.pyplot as plt
import gc

import hydra
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from tqdm import tqdm
from omegaconf import DictConfig

from src import utils, loss_tracker, bp_model
import gpustat

def train(itr: int, opt, model, optimizer, mnist=True):

    print("MNIST") if mnist else print("CIFAR10")
    
    
    start_time = time.time()
    train_loader = utils.get_data(opt, "train", mnist)
    num_steps_per_epoch = len(train_loader)

    loss_track = loss_tracker.LossTracker(opt.model.num_layers)
    
    memory_usage = []
    power_usage = []
    util_usage = []
    val_acc = []

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
            
        
        gpu_query = gpustat.GPUStatCollection.new_query()
        memory_usage.append(gpu_query[0].memory_used)
        # print(f"Memory usage: {memory_usage}")
        
        power_usage.append(gpu_query[0].power_draw)
        # print(f"Power usage: {power_usage}")
        
        util_usage.append(gpu_query[0].utilization)
        # print(f"Utilization: {util_usage}")
               
        # Record loss 
        loss_track.update(train_results, epoch)

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            acc = validate_or_test(opt, model, "val", mnist, epoch=epoch)
            val_acc.append(acc)

        if acc > 0.97:
            epochs = epoch + 1
            break
    # Creating and saving a dataframe of memory, power and utilization at each epoch
    epoch_data = {"Epoch": range(epochs),
                  "Memory": memory_usage,
                  "Power": power_usage,
                  "Utilization": util_usage,
                  "Val_Acc": val_acc}
    pd.DataFrame(epoch_data).to_csv(f"./Outputs/FF_epoch_data.csv", index=False)
     
    # Save loss plot at end of training
    loss_track.plot()

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

    return (test_results["classification_accuracy"])


@hydra.main(config_path=".", config_name="config", version_base=None)
def ff_main(opt: DictConfig) -> None:
    
    # True = mnist, False = cifar10
    mnist_T_cifar_F = True
    
    opt = utils.parse_args(opt)
    model, optimizer = utils.get_model_and_optimizer(opt, "mnist") # mnist/cifar10
    model = train(itr, opt, model, optimizer, mnist_T_cifar_F) 
    validate_or_test(opt, model, "val", mnist_T_cifar_F)  
    
    # Extracting network parameters
    print(f"Network parameters: {model.parameters()}")

    if opt.training.final_test:
        validate_or_test(opt, model, "test", mnist_T_cifar_F) 
    

@hydra.main(config_path=".", config_name="config", version_base=None)
def bp_main(opt: DictConfig) -> None:
    print(f'BP START')
    device = opt.device

    print(f'Using {device}')
    
    torch.manual_seed(opt.seed)
    train_loader, val_loader, test_loader = utils.MNIST_loaders(
        train_batch_size=opt.input.batch_size, 
        test_batch_size=opt.input.batch_size
    )

    dims = [784, 1000, 1000, 1000]
    num_classes = 10

    bp_net = bp_model.BPNet(dims, device, num_classes)
    bp_net.to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.Adam(bp_net.parameters(), lr=opt.training.learning_rate)

    # Switch depending on training experiment
    epochs = opt.training.epochs          # same epoch
    # epochs = 50                            # independent
    
    # Extracting network parameters
    print(f"Network parameters: {bp_net.parameters()}")
    
     
    train_losses = []
    val_losses = []
    val_accs = []
    epoch_timestamps = []
     
    memory_usage = []
    power_usage = []
    util_usage = []



    for epoch in tqdm(range(epochs)):
        bp_start_time = time.time()
        train_loss = utils.train(bp_net, device, train_loader, optim, loss_fn, epoch)
        val_loss, val_accuracy = utils.evaluate(bp_net, device, val_loader, loss_fn, True)
                
        gpu_query = gpustat.GPUStatCollection.new_query()
        memory_usage.append(gpu_query[0].memory_used)
        # print(f"Memory usage: {memory_usage}")
        
        power_usage.append(gpu_query[0].power_draw)
        # print(f"Power usage: {power_usage}")
        
        util_usage.append(gpu_query[0].utilization)
        # print(f"Utilization: {util_usage}")
               
 
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        print(f'Time: {time.time() - bp_start_time}')
        print()
        epoch_timestamps.append((epoch, time.time() - bp_start_time))
        
        if val_accuracy > 0.97:
            epochs = epoch + 1
            break
    
    print(len(range(epochs)))
    print(len(memory_usage))
    print(len(power_usage))
    print(len(util_usage))
    print(len(val_accs))
    
    epoch_data = {"Epoch": range(epochs),
                  "Memory": memory_usage,
                  "Power": power_usage,
                  "Utilization": util_usage,
                  "Val_Acc": val_accs}
    pd.DataFrame.from_dict(epoch_data).to_csv(f"./Outputs/BP_epoch_data.csv", index=False)

    # Test on full trained net
    test_loss, test_accuracy = utils.evaluate(bp_net, device, test_loader, loss_fn)
   

    # plot BP training time per epoch
    ep , t = zip(*epoch_timestamps)

    plt.figure(figsize=(10, 6))
    plt.plot(ep,t)
    plt.title('BP Training time per Epoch')
    plt.xlabel('Epoch')
    # plt.ylim(ymin=0)
    plt.ylabel('Runtime')
    plt.savefig('./images/bp_tpe.png')
    plt.close()

    # plot BP loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Backpropagation Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(f'./images/bp_loss_curves.png')
    plt.close()

    # Clear and delete model from cuda
    del bp_net
    del optim
    del loss_fn
    torch.cuda.empty_cache()
    gc.collect() # free up memory


if __name__ == "__main__":
    
    for itr in range(10):
        print(f'Iteration: {itr}')
        ## Define logging queues
        power_log_queue = queue.Queue()
        util_log_queue = queue.Queue()
        memory_log_queue = queue.Queue()

        ## Define stop event
        stop_event = threading.Event()

        ## Define logging threads
        power_logging_thread = threading.Thread(
            target=utils.log_gpu_power, args=(stop_event, power_log_queue)
        )
        util_logging_thread = threading.Thread(
            target=utils.log_gpu_util, args=(stop_event, util_log_queue)
        )
        mem_logging_thread = threading.Thread(
            target=utils.log_gpu_mem, args=(stop_event, memory_log_queue)
        )

        ## Start logging threads
        power_logging_thread.start()
        util_logging_thread.start()
        mem_logging_thread.start()

        ##### RUN BP #####
        BP_start_time = time.time()
        bp_main()
        BP_end_time = time.time()


        print()
        torch.cuda.empty_cache()
        time.sleep(10)

        ##### RUN FF algorithm #####
        FF_start_time = time.time()
        ff_main()
        FF_end_time = time.time()
    

        stop_event.set()

        mem_logging_thread.join()
        util_logging_thread.join()
        power_logging_thread.join()

        ## Extracting logs
        power_log = []
        while not power_log_queue.empty():
            power_log.append(power_log_queue.get())

        util_log = []
        while not util_log_queue.empty():
            util_log.append(util_log_queue.get())

        memory_log = []
        while not memory_log_queue.empty():
            memory_log.append(memory_log_queue.get())

        ## Extracting timestamps
        power_timestamps = [x[0] for x in power_log]
        power_values = [x[1] for x in power_log]

        util_timestamps = [x[0] for x in util_log]
        util_values = [x[1] for x in util_log]

        memory_timestamps = [x[0] for x in memory_log]
        memory_values = [x[1] for x in memory_log]

        ## Saving log data as CSV
        pd.DataFrame(power_log, columns=["Timestamp", "Value"]).to_csv(f"./Outputs/power_log_{itr}.csv" , index = False)
        pd.DataFrame(util_log, columns=["Timestamp", "Value"]).to_csv(f"./Outputs/util_log_{itr}.csv", index=False)
        pd.DataFrame(memory_log, columns=["Timestamp", "Value"]).to_csv(f"./Outputs/memory_log_{itr}.csv", index=False)

        ## Saving FF and BP timestamps as CSV
        model_timestamps = {"BP": [BP_start_time, BP_end_time],
                            "FF": [FF_start_time, FF_end_time]}
        pd.DataFrame(model_timestamps).to_csv(f"./Outputs/model_timestamps_{itr}.csv", index = False)



