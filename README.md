

# Pruning

## MNIST
| Network              | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|----------------------|-------------|-------------|------------|------------------|
| LeNet-300-100 Ref    | 1.30%       | -           | 267K       | -                |
| LeNet-300-100 Pruned | 1.40%       | -           | **22K**    | **12X**          |
| LeNet-5 Ref          | 0.99%       | -           | 431K       | -                |
| LeNet-5 Pruned       | 0.93%       | -           | **36K**    | **12X**          |

## CIFAR-100

| Network        | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|----------------|-------------|-------------|------------|------------------|
| LeNet-5 Ref    | 61.17%      | 31.55%      | 431K       | -                |
| LeNet-5 Pruned |             | -           |            |                  |

## Imagenette
| Network              | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|----------------------|-------------|-------------|------------|------------------|
| AlexNet Ref          | 20.80%      | 3.33%       | 61M        | -                |
| AlexNet Pruned       |             |             |            |                  |
| VGG16 Ref            | 20.80%      | 3.33%       | 61M        | -                |
| VGG16 Pruned         |             |             |            |                  |

# Quantization



# Acknowledgments
- [Pytorch](https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils) for pruning library
- [Distiller](https://github.com/IntelLabs/distiller) for sensitivity analysis
- [pytorch-template](https://github.com/victoresque/pytorch-template) for project structure