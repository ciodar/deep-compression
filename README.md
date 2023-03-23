# Deep Compression

![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)

This repository contains an unofficial [Pytorch Lightning](https://lightning.ai/pages/open-source/) implementation of the paper "**Deep Compression**: Compressing Deep Neural Networks with pruning,trained quantization and Huffman coding"
by Song Han et al. (https://arxiv.org/abs/1510.00149).
It provides an implementation of the three core methods described in the paper:
- Pruning
- Quantization
- Huffman Encoding

This project was implemented by **Dario Cioni** (7073911) for **Deep Learning** exam at University of Florence.

## Table of contents


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
  - tqdm
  - tensorboard

## Project Structure
  ```
  model-compression/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model 
  │
  ├── configs/ - directory of saved model configurations for various datasets and models
  ├── config.json - default working configuration file for training
  ├── parse_config.py - handles config file and cli options
  │
  ├── data.py - anything about data loading goes here
  │   ├── BaseDataLoader - Abstract Base Class for Dataloader
  │   ├── MnistDataLoader - DataLoader for MNIST
  │   ├── CIFAR100DataLoader - DataLoader for CIFAR 100
  │   └── ImagenetteDataLoader - DataLoader for Imagenette
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
  │   │   ├── IterativePruning - Custom callback extending ModulePruning allowing 
  │   │   └── Quantization - Custom callback defining quantization process
  │   │
  │   ├── lit_model.py - Lightning wrapper for model training
  │   ├── metrics.py - code to define metrics
  │   └── trainer.py - code to configure a Lightning Trainer from json configuration 
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils.py -  utility functions
  ```

## Usage
### Training
To train a new model from scratch
```sh
$ python train.py -c config.json
```

To resume a training
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

- `pruning_fn`:  Function from torch.nn.utils.prune module or a PyTorch BasePruningMethod subclass. Can also be string e.g. “l1_unstructured”
- `parameter_names`: List of parameter names to be pruned from the nn.Module. Can either be "weight" or "bias".
- `parameters_to_prune`: List of tuples (nn.Module, "parameter_name_string"). If unspecified, retrieves all module in model having `parameter_names`.
- `use_global_unstructured`: Whether to apply pruning globally on the model. If parameters_to_prune is provided, global unstructured will be restricted on them.
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


### MNIST
| Network                                    | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|--------------------------------------------|-------------|-------------|------------|------------------|
| LeNet-300-100 Ref                          | 1.57%       | -           | 267K       | -                |
| LeNet-300-100 One-Shot Pruning w/ retrain  | 1.60%       | -           | **22K**    | **12X**          |
| LeNet-300-100 Iterative Pruning w/ retrain | 1.58%       | -           | **22K**    | **12X**          |
| LeNet-5 Ref                                | 0.7%        | -           | 429K       | -                |
| LeNet-5 One-Shot Pruning w/ retrain        | 0.7%        | -           | **36K**    | **12X**          |
| LeNet-5 Iterative Pruning w/ retrain       | 0.7%        | -           | **36K**    | **12X**          |

### CIFAR-100

| Network        | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|----------------|-------------|-------------|------------|------------------|
| LeNet-5 Ref    | 61.17%      | 31.55%      | 266K       | -                |
| LeNet-5 Pruned | 61.89%      | 31.87%      | **50K**    | **11X**          |

### Imagenette
| Network              | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|----------------------|-------------|-------------|------------|------------------|
| AlexNet Ref          | 19.87%      | -           | 57M        | -                |
| AlexNet Pruned       | 20.82%      | -           | 9M         | **8X**           |
| VGG16 Ref            | -           | -           | 61M        | -                |
| VGG16 Pruned         |             | -           |            |                  |

## Quantization
Quantization is implemented with `Quantizer`, a custom callback called by Pytorch Lightning's [Trainer](https://lightning.ai/docs/pytorch/latest/common/trainer.html).

The `Quantizer` callback implements abstract [Callback](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html#lightning.pytorch.callbacks.Callback) class
and allows to run vector quantization on Linear and Conv2d modules.

Vector quantization is implemented in `BaseQuantizationMethod` class, a novel module inspired on existing pruning pipeline in torch.nn.utils.prune.
This module takes care of performing a clustering of a parameter's weights and store it as a tensor of cluster centers and an index matrix.
The initialization of the cluster centroids can be done in three different ways

- Linear: choose linearly-spaced values between the minimum and maximum of weight magnitude
- Density-based: 
- Random/Forgy: randomly chooses values from the weight matrix.

The callback calls the quantization function for each layer


| Network | Quantization type | Top-1 Error | Top-5 Error |
|---------|-------------------|-------------|-------------|
| LeNet-5 | Forgy             | 61.27%      | 30.25%      |
| LeNet-5 | Density-based     |             |             |
| LeNet-5 | Linear            | -           | -           |
| AlexNet | Forgy             |             |             |
| AlexNet | Density-based     |             |             |
| AlexNet | Linear            | -           | -           |
| VGG-16  | Forgy             |             |             |
| VGG-16  | Density-based     |             |             |
| VGG-16  | Linear            | -           | -           |

## Huffman encoding
Work in progress

# Acknowledgments
- [Pytorch](https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils) for pruning library
- [Distiller](https://github.com/IntelLabs/distiller) for sensitivity analysis
- [pytorch-template](https://github.com/victoresque/pytorch-template) for project structure