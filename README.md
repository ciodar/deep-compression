

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
| LeNet-5 Pruned | 61.89%      | 31.87%      | **50K**    | **11X**          |

## Imagenette
| Network              | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|----------------------|-------------|-------------|------------|------------------|
| AlexNet Ref          | 20.80%      | 3.33%       | 61M        | -                |
| AlexNet Pruned       |             |             |            |                  |
| VGG16 Ref            | -           | -           | 61M        | -                |
| VGG16 Pruned         |             |             |            |                  |

# Quantization

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