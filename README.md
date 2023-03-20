# Deep Compression


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
  ├── runs/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   ├── lit_model.py - Lightning wrapper for models
  │   ├── metrics.py - Trainer for initial training
  │   └── trainer.py - Custom Trainer for Deep Compression pipeline
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils.py -  utility functions
  ```


## Pruning

### MNIST
| Network              | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|----------------------|-------------|-------------|------------|------------------|
| LeNet-300-100 Ref    | 1.30%       | -           | 267K       | -                |
| LeNet-300-100 Pruned | 1.40%       | -           | **22K**    | **12X**          |
| LeNet-5 Ref          | 0.99%       | -           | 431K       | -                |
| LeNet-5 Pruned       | 0.93%       | -           | **36K**    | **12X**          |

### CIFAR-100

| Network        | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|----------------|-------------|-------------|------------|------------------|
| LeNet-5 Ref    | 61.17%      | 31.55%      | 431K       | -                |
| LeNet-5 Pruned | 61.89%      | 31.87%      | **50K**    | **11X**          |

### Imagenette
| Network              | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|----------------------|-------------|-------------|------------|------------------|
| AlexNet Ref          | 19.87%      | -           | 57M        | -                |
| AlexNet Pruned       | 20.82%      | -           | 9M         | **8X**           |
| VGG16 Ref            | -           | -           | 61M        | -                |
| VGG16 Pruned         |             | -           |            |                  |

## Quantization

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