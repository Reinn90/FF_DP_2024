# Western Sydney University Bachelor of Data Science Discovery Project 2024 - Evaluating the Efficiency of the Forward Forward Algorithm
Project members:
- Kevin Reyes
- Gabriel Schussler

## Abstract
This research project examines the computational efficiency of Geoffrey Hinton’s
Forward-Forward algorithm relative to traditional backpropagation in artificial neural network training. The Forward-Forward (FF) algorithm replaces the usual
forward-backward cycle with two forward passes using positive and negative data
samples, which may improve training speed and resource efficiency. We implemented
both the FF and backpropagation (BP) algorithms in PyTorch, comparing their performance on the MNIST dataset in terms of training time, GPU memory usage, GPU
utilisation, and power consumption.
At a target accuracy of 97%, the FF algorith reached this thresold approximately
50 seconds faster than BP and uses 20% less total power. Additionally, the FF
method showed reduced GPU utilisation (27.39% vs 36.03%), indicating potential
benefits in resource-limited settings, though memory usage was similar across methods. Although this study faced limitations related to hardware and implementation,
the results suggest that the FF algorithm may provide valuable efficiency advantages, particularly in energy-sensitive contexts. Future research could explore the
algorithm’s performance across other datasets, various network architectures, and
applications beyond computer vision.





Majority of the NN model code is take from the repo below, and augmented/repurposed for our project
***
# Reimplementation of the Forward-Forward Algorithm

This is a reimplementation of Geoffrey Hinton's Forward-Forward Algorithm in Python/Pytorch.

&rarr; [Original Paper](https://arxiv.org/abs/2212.13345)

&rarr; [Official Matlab Implementation](https://www.cs.toronto.edu/~hinton/)

This code covers the experiments described in section 3.3 ("A simple supervised example of FF") of the paper and 
achieves roughly the same performance as the official Matlab implementation (see Results section).


## The Forward-Forward Algorithm

The Forward-Forward algorithm is a method for training deep neural networks in a more biologically plausible manner.
Instead of sharing gradients between layers, it trains each layer based on local losses. 

To implement these local losses, the network performs two forward passes:
The first forward pass is on positive samples, which are representative of the "real" data. 
For these samples, the network is trained to maximize the "goodness" for each of its layers. 
In the second forward pass, the network is fed negative samples, 
which are data perturbations that do not conform to the true data distribution. 
For these samples, the network is trained to minimize the "goodness".

The goodness can be evaluated in several ways, such as by taking the sum of the squared activities of a layer.

<img src="images/ForwardForward.jpeg" alt="The Forward-Forward Algorithm" width="600"/>

The image above depicts the training of a network with the Forward-Forward algorithm as implemented in this repository. 
Here, the positive and negative samples are created by adding a one-hot encoding of the correct or incorrect label 
to the first ten pixels of the image.


## How to Use

### Setup
- Install [conda](https://www.anaconda.com/products/distribution)
- Adjust the ```setup_conda_env.sh``` script to your needs (e.g. by setting the right CUDA version)
- Run the setup script:
```bash
bash setup_conda_env.sh
```


### Run Experiments
- Run the training and evaluation with forward-forward:
```bash
source activate FF
python -m main
```


## Results
Comparison of the results for different implementations of the Forward-Forward algorithm:

| | Test Error (%) |
| --- | -- |
| Paper | 1.36 |
| Official Matlab Implementation | 1.47 |
| This Repo | 1.45 |
