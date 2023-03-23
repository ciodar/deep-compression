# Deep Compression

![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)

This repository contains an unofficial [Pytorch Lightning](https://lightning.ai/pages/open-source/) implementation of the paper "Deep Compression: Compressing Deep Neural Networks with pruning,trained quantization and Huffman coding"
by Song Han et al. (https://arxiv.org/abs/1510.00149).

It provides an implementation of the three core methods described in the paper:
- Pruning
- Quantization
- Huffman Encoding

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
  │   ├── alexnet.py - Implementation of AlexNet
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
Pruning is implemented as a callback for Pytorch Lightning's `Trainer`.

The `IterativePruning` callback in trainer/callbacks/pruning.py  extends `ModelPruning` callback with further control on the pruning schedule.

- epochs:
- start_epoch:
- prune_every:
- 

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
Quantization is implemented with `Quantizer` callback

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





# Acknowledgments
- [Pytorch](https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils) for pruning library
- [Distiller](https://github.com/IntelLabs/distiller) for sensitivity analysis
- [pytorch-template](https://github.com/victoresque/pytorch-template) for project structure