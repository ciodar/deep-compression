[![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-blueviolet)](#)

# Deep Compression

This repository is an unofficial [Pytorch Lightning](https://lightning.ai/pages/open-source/) 
implementation of the paper "**Deep Compression**: Compressing Deep Neural Networks with pruning,trained quantization and Huffman coding" by Song Han et al. (https://arxiv.org/abs/1510.00149).
It provides an implementation of the three core methods described in the paper:

- Pruning
- Quantization
- Huffman Encoding

These are the main results on the MNIST and [Imagenenette](https://github.com/fastai/imagenette) datasets

| Network                  | Top-1 Error (Ours) | Top-1 Error (Han et al.) | Compression Rate (Ours) | Compression Rate (Han et al.) |
|--------------------------|--------------------|--------------------------|-------------------------|-------------------------------|
| LeNet-300-100 Ref        | 2.0%               | 1.64%                    | -                       | -                             |
| LeNet-300-100 Compressed | 1.8%               | 1.58%                    | **48X**                 | 40X                           |
| LeNet-5 Ref              | 0.83%              | 0.8%                     | -                       | -                             |
| LeNet-5 Compressed       | 0.74%              | 0.74%                    | **46X**                 | 39X                           |
| AlexNet Ref              | 9.11%              | -                        | -                       | -                             |
| AlexNet Compressed       | 3.87%              | -                        | **41X**                 | 35X                           |

This project was implemented by **Dario Cioni** (7073911) for **Deep Learning** exam at University of Florence.

## Table of contents

<!-- TOC -->
* [Deep Compression](#deep-compression)
  * [Table of contents](#table-of-contents)
  * [Requirements](#requirements)
  * [Project Structure](#project-structure)
  * [Usage](#usage)
    * [Models](#models)
    * [Configuration file](#configuration-file)
    * [Training](#training)
    * [Testing](#testing)
    * [Sensitivity analysis](#sensitivity-analysis)
  * [Pruning](#pruning)
  * [Quantization](#quantization)
  * [Huffman encoding](#huffman-encoding)
  * [Results](#results)
    * [MNIST](#mnist)
    * [Imagenette](#imagenette)
  * [References](#references)
  * [Acknowledgments](#acknowledgments)
<!-- TOC -->

## Requirements
  - pytorch
  - pytorch-lightning
  - torchmetrics
  - torchvision
  - ipykernel
  - jupyter
  - matplotlib
  - numpy
  - scipy
  - scikit-learn
  - tqdm
  - tensorboard

## Project Structure
  ```
  model-compression/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model 
  │
  ├── compression/ - directory containing all the Deep Compression logic
  │   ├── pruning.py - implements ThresholdPruning and utilities for sparsity calculation
  │   ├── quantization.py - implements all the weight sharing logic, utilities for compression calculation
  │   └── huffman_encoding.py - implements huffman encoding
  │
  ├── configs/ - directory of saved model configurations for various datasets and models
  ├── config.json - a configuration file for your current experiment. 
  ├── parse_config.py - handles config file and cli options
  │
  ├── data.py - anything about data loading goes here
  │   ├── BaseDataLoader - Abstract Base Class for Dataloader
  │   ├── MnistDataLoader - DataLoader for MNIST
  │   ├── CIFAR100DataLoader - DataLoader for CIFAR 100
  │   └── ImagenetDataLoader - DataLoader for Imagenet-like datasets
  │
  ├── data/ - directory for storing input data
  │
  ├── models/ - directory of developed models
  │   ├── lenet.py - Implementation of LeNet300-100 and LeNet-5
  │   ├── alexnet.py - Implementation of AlexNet which follows Caffe implementation 
  │   │                https://github.com/songhan/Deep-Compression-AlexNet/blob/master/bvlc_alexnet_deploy.prototxt
  │   └── vgg.py - Implementation of VGG-16
  │
  ├── notebooks/ - directory containing example notebooks 
  │   ├── mnist-lenet300.ipynb - Deep Compression pipeline example on MNIST with LeNet-300-100 FC model
  │   ├── mnist-lenet5.ipynb - Deep Compression pipeline example on MNIST with LeNet-5 model
  │   └── ...
  │
  ├── runs/ - trained models and logs are saved here
  │
  ├── trainer/ - module containing code for training and evaluating models
  │   ├── callbacks/ - module containing custom callbacks for Lightning Trainer
  │   │   ├── IterativePruning - Custom callback extending ModulePruning allowing finegraned control on the pruning process
  │   │   └── Quantization - Custom callback defining quantization process. Also handles huffman encoding calculation.
  │   │
  │   ├── lit_model.py - Lightning wrapper for model training
  │   ├── metrics.py - code to define metrics
  │   └── trainer.py - code to configure a Lightning Trainer from json configuration 
  │
  ├── logger/ - module for additional console logging (Tensorboard is handled by Lightning)
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils.py -  utility functions
  ```

## Usage

### Models
[models](models) folder contains the implementation of the following models:

- LeNet-300 from the original [LeNet paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- LeNet-5, in a modified, larger version which follows the one in the [Deep Compression paper](https://arxiv.org/abs/1510.00149)
- AlexNet, which follows the Caffe implementation available the author's [repository](https://github.com/songhan/Deep-Compression-AlexNet)
- VGG-16 

### Configuration file
All the experiments are handled by a configuration file in `.json` format:

````json
{
    "name": "Mnist_LeNet300",
    "n_gpu": 1,
    "arch": {
        "type": "LeNet300",
        "args": {
            "num_classes": 10,
            "grayscale": true,
            "dropout_rate": 0
        }
    },
    "data_loader": {
        "type": "MnistDataLoader",
        "args": {
            "data_dir": "data/",
            "batch_size": 128,
            "shuffle": true,
            "validation_split": 0.1,
            "num_workers": 6,
            "resize": false
        }
    },
    "optimizer": {
        "type": "SGD",
        "args": {
            "lr": 1e-2,
            "momentum": 0.9,
            "weight_decay": 1e-3,
            "nesterov": true
        }
    },
    "loss": "cross_entropy",
    "metrics": [
        "accuracy",
        "topk_accuracy"
    ],
    "trainer": {
        "min_epochs": 10,
        "max_epochs": 20,
        "save_dir": "runs/",
        "verbosity": 1,
        "monitor": "max val_accuracy",
        "loggers": ["TensorBoard"],
        "callbacks": {
            "ModelCheckpoint": {
                "save_top_k": 1,
                "monitor": "val_accuracy",
                "every_n_epochs":5,
                "mode": "max"
            },
            "IterativePruning": {
                "pruning_schedule": {
                    "target_sparsity": 0.9,
                    "start_epoch": 0,
                    "prune_every": 2
                },
                "pruning_fn": "l1_threshold",
                "parameter_names": ["weight"],
                "amount": 0.6,
                "use_global_unstructured": true,
                "make_pruning_permanent": false,
                "verbose": 2
            },
            "Quantization": {
              "epoch": 10,
              "quantization_fn": "linear_quantization",
              "parameter_names": ["weight"],
              "filter_layers": ["Linear"],
              "bits": 6,
              "verbose": 2,
              "huffman_encode": true
            }
        }
    }
}


````
### Training
To train a new model from scratch, use the command -c or --config followed by the path to a JSON configuration file
```sh
$ python train.py -c config.json
```

To resume a training, use the command -r followed by the path to a Pytorch Lightning checkpoint. 

In the same directory it should also be placed the JSON configuration file of the trained model.
This is useful if you want to perform compression step by step, changing the callbacks every time.
```sh
$ python train.py -r path-to-checkpoint/checkpoint.ckpt
```

### Testing
```sh
$ python test.py -r path-to-checkpoint/checkpoint.ckpt
```

### Sensitivity analysis

```sh
$ python sensitivity.py -r path-to-checkpoint/checkpoint.ckpt
```



## Pruning
Pruning is implemented as a callback, called during training by Pytorch Lightning's [Trainer](https://lightning.ai/docs/pytorch/latest/common/trainer.html).
The `IterativePruning` callback extends [ModelPruning](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelPruning.html#lightning.pytorch.callbacks.ModelPruning) callback with further control on the pruning schedule.
It allows to set a target sparsity level, prune each layer with a different amount/threshold and perform Iterative pruning.

- `pruning_fn`:  Function from torch.nn.utils.prune module or a PyTorch BasePruningMethod subclass. Can also be string e.g. “l1_unstructured”
- `parameter_names`: List of parameter names to be pruned from the nn.Module. Can either be "weight" or "bias".
- `parameters_to_prune`: List of tuples (nn.Module, "parameter_name_string"). If unspecified, retrieves all module in model having `parameter_names`.
- `use_global_unstructured`: Whether to apply pruning globally on the model. If `parameters_to_prune` is provided, global unstructured will be restricted on them.
- `amount`: Quantity of parameters to prune. Can either be
  - int specifying the exact amount of parameter to prune
  - float specifying the percentage of parameters to prune
  - List of int or float speciying the amount to prune in each module. The length of  This is allowed only if `use_global_unstructured=False`
- `filter_layers`: List of strings, filters pruning only on layers of a specific class ("Linear","Conv2d" or both.)

The `pruning_schedule` is provided as a dictionary in trainer's JSON configuration and allows the following arguments:
- `epochs`: list specifying the exact epochs in which pruning is performed. If specified, overrides `start_epoch` and `prune_every`
- `start_epoch`: first epoch in which pruning is performed. 
- `prune_every`: performs pruning every `prune_every` epochs. Default=1.
- `target_sparsity`: prevents from applying pruning if the model's sparsity is greater than `target_sparsity`

Performance of pruned models was evaluated on different datasets in different settings
- One-shot pruning with retraining: prune a trained model, then retrain the weights to compensate the accuracy loss occurred during pruning
- Iterative pruning: iteratively prune and retrain the model multiple times

| Network                                   | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|-------------------------------------------|-------------|-------------|------------|------------------|
| LeNet-300-100 Ref                         | 2.0%        | -           | 267K       | -                |
| LeNet-300-100 one-shot pruning w/ retrain | 1.83%       | -           | **22K**    | **12X**          |
| LeNet-5 Ref                               | 0.83%       | -           | 429K       | -                |
| LeNet-5 one-shot pruning w/ retrain       | 0.69%       | -           | **36K**    | **12X**          |
| AlexNet Ref                               | 9.11%       | -           | 57M        | -                |
| AlexNet one-shot pruning w/ retrain       | 2.627%      | -           | 6M         | **10X**          |
| VGG16 Ref                                 | -           | -           | 61M        | -                |
| VGG16 Pruned                              |             | -           |            |                  |

## Quantization
Quantization is implemented with `Quantizer`, a custom callback called by Pytorch Lightning's [Trainer](https://lightning.ai/docs/pytorch/latest/common/trainer.html).

The `Quantizer` callback implements abstract [Callback](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html#lightning.pytorch.callbacks.Callback) class
and allows to run vector quantization on Linear and Conv2d modules.

Vector quantization is implemented in `BaseQuantizationMethod` class, a novel module inspired on existing pruning pipeline in torch.nn.utils.prune.
This module takes care of performing a clustering of parameter's weights and store it as a tensor of cluster centers and an index matrix.

The initialization of the cluster centroids can be done in three different ways

- **Linear**: choose linearly-spaced values between [ _min_ , _max_ ] of the original weights
- **Density-based**: chooses the weights based on the density distribution. It linearly spaces the CDF of the weights in the y-axis, then finds the horizontal intersection with the CDF, and finally the  vertical intersection on the x-axis, which becomes the centroid. 
- **Random/Forgy**: randomly chooses _k_ weights from the weight matrix.

The callback calls the quantization function for each layer and accepts the following parameters:
- `epoch`: an int indicating the epoch on which quantization is performed
- `quantization_fn`: Function from [compression.quantization](compression/quantization.py) module, passed as a string. Available functions are: "density_quantization","linear_quantization","forgy_quantization"
- `parameter_names`: List of parameter names to be quantized from the nn.Module. Can either be "weight" or "bias".
- `filter_layers`: List of strings, filters pruning only on layers of a specific class ("Linear","Conv2d" or both.)
- `bits`: an int indicating the number of bits used for quantization. The number of codebook weights will be 2**bits.

| #CONV bits / #FC bits | Top-1 Error | Top-1 Error increase | Compression rate |
|-----------------------|-------------|----------------------|------------------|
| 32 bits / 32 bits     | 2.627%      | -                    | -                |
| 8 bits / 5 bits       | 3.87%       | 1.2%                 | 41X              |
| 8 bits / 4 bits       | 4.066%      | 1.4%                 | 44X              |
| 4 bits / 2 bits       | 3.45%       | 0.8%                 | 61X              |

## Huffman encoding
Huffman Encoding is implemented in [compression.huffman_encoding](compression/huffman_encoding.py) model.

This module computes the huffman tree for the passed vector and calculates the memory saving obtained by that encoding and the average number of bits used to encode every element of the vector.
The encoding is not actually applied to the vector.

Huffman Encoding is enabled by setting the parameter `huffman_encode` to True in `Quantization` callback.

## Results

Here's a summary of the reached compression of each model, after pruning, quantization and Huffman Encoding. 
The experiments are available on Tensorboard.dev.

### MNIST

- [LeNet-300-100](https://tensorboard.dev/experiment/Z7XtG6YXRdOlBaX9Ramt3g/)

| Layer     | # Weights | Weights % (P) | Weight bits (P+Q) | Weight bits (P+Q+H) | Index bits (P+Q) | Index bits (P+Q+H) | Compress rate (P+Q) | Compress rate (P+Q+H) |
|-----------|-----------|---------------|-------------------|---------------------|------------------|--------------------|---------------------|-----------------------|
| fc1       | 235K      | 8%            | 6                 | 5.1                 | 5                | 2.5                | 2.53%               | 1.92%                 |
| fc2       | 30K       | 9%            | 6                 | 5.4                 | 5                | 3.6                | 3.03%               | 2.71%                 |
| fc3       | 1K        | 26%           | 6                 | 5.8                 | 5                | 3.1                | 14.52%              | 13.59%                |
| **Total** | 266K      | 8% (12X)      | 6                 |                     | 5                |                    | 2.63% (38.0X)       | 2.05% (48.7X)         |

- [LeNet-5](https://tensorboard.dev/experiment/RMyp5qxRRSyZP0zn4Oe7wA/)

| Layer     | # Weights | Weights % (P) | Weight bits (P+Q) | Weight bits (P+Q+H) | Index bits (P+Q) | Index bits (P+Q+H) | Compress rate (P+Q) | Compress rate (P+Q+H) |
|-----------|-----------|---------------|-------------------|---------------------|------------------|--------------------|---------------------|-----------------------|
| conv1     | 0.5K      | 82%           | 8                 | 7.9                 | 5                | 1.2                | 92.47%              | 74.54%                |
| conv2     | 25K       | 19%           | 8                 | 7.5                 | 5                | 3.0                | 21.10%              | 7.09%                 |
| fc1       | 400K      | 7%            | 5                 | 4.2                 | 5                | 3.6                | 1.97%               | 1.66%                 |
| fc2       | 3K        | 73%           | 5                 | 4.4                 | 5                | 1.4                | 21.58%              | 14.08%                |
| **Total** | 429K      | 8% (12X)      |                   |                     | 5                |                    | 3.34% (39X)         | 2.15% (46X)           |


### Imagenette

- [AlexNet](https://tensorboard.dev/experiment/2xJrx1AYRAK4WiobyudX2Q/)

| Layer     | # Weights | Weights % (P) | Weight bits (P+Q) | Weight bits (P+Q+H) | Index bits (P+Q) | Index bits (P+Q+H) | Compress rate (P+Q) | Compress rate (P+Q+H) |
|-----------|-----------|---------------|-------------------|---------------------|------------------|--------------------|---------------------|-----------------------|
| conv1     | 35K       | 84%           | 8                 | 7.2                 | 5                | 1.2                | 32.6%               | 23.61%                |
| conv2     | 307K      | 38%           | 8                 | 6.8                 | 5                | 2.6                | 14.33%              | 11.20%                |
| conv3     | 885K      | 35%           | 8                 | 6.5                 | 5                | 2.7                | 13.16%              | 10.13%                |
| conv4     | 663K      | 37%           | 8                 | 6.6                 | 5                | 2.7                | 13.9%               | 10.96%                |
| conv5     | 442K      | 37%           | 8                 | 6.7                 | 5                | 2.7                | 13.92%              | 11.06%                |
| fc1       | 38M       | 9%            | 5                 | 4.0                 | 5                | 4.5                | 2.53%               | 2.07%                 |
| fc2       | 17M       | 9%            | 5                 | 4.1                 | 5                | 4.6                | 2.53%               | 1.99%                 |
| fc3       | 4M        | 25%           | 5                 | 4.4                 | 5                | 3.3                | 7.11%               | 5.95%                 |
| **Total** | 58M       | 11% (10X)     | 5.4               |                     | 5                |                    | 3.03%  (32X)        | 2.43% (41X)           |

## References
[[1]](https://arxiv.org/pdf/1510.00149v5.pdf) Han, Song, Huizi Mao, and William J. Dally. "Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding." arXiv preprint arXiv:1510.00149 (2015)

[[2]](https://arxiv.org/pdf/1506.02626v3.pdf) Han, Song, et al. "Learning both weights and connections for efficient neural network." Advances in neural information processing systems 28 (2015)

## Acknowledgments
- [Pytorch](https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils) for pruning library
- [Distiller](https://github.com/IntelLabs/distiller) for sensitivity analysis
- [pytorch-template](https://github.com/victoresque/pytorch-template) for project structure and experiment logging