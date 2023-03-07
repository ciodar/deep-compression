

# Pruning

| Network              | Top-1 Error | Top-5 Error | Parameters | Compression Rate |
|----------------------|-------------|-------------|------------|------------------|
| LeNet-300-100 Ref    | 2.00%       | -           | 267K       | -                |
| LeNet-300-100 Pruned | 1.30%       | -           | **22K**    | **12X**          |
| LeNet-5 Ref          |             |             | 431K       | -                |
| LeNet-5 Pruned       |             |             |            |                  |
| AlexNet Ref          | 20.80%      | 3.33%       | 61M        | -                |
| AlexNet Pruned       |             |             |            |                  |

# Quantization



# Acknowledgments
- [Pytorch](https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils) for pruning library
- [Distiller](https://github.com/IntelLabs/distiller) for sensitivity analysis
- [pytorch-template](https://github.com/victoresque/pytorch-template) for project structure