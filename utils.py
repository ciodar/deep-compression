import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


# functions to show an image
def imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_sparsity_matrix(model):
    #fig = plt.figure()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weights = module.weight.detach().cpu()
            plt.spy(weights, color='blue', markersize=1)
            plt.title(name)
            plt.show()
        # if isinstance(param, torch.nn.Conv2d):
        elif isinstance(module, torch.nn.Conv2d):
            weights = module.weight.detach().cpu()
            num_kernels = weights.shape[0]
            for k in range(num_kernels):
                kernel_weights = weights[k].sum(dim=0)
                tag = f"{name}/kernel_{k}"
                plt.spy(kernel_weights, color='blue', markersize=1)
                plt.title(tag)
                plt.show()

                # ax = fig.add_subplot(1, num_kernels, k + 1, xticks=[], yticks=[])
                # ax.set_title("layer {0}/kernel_{1}".format(name, k))
    #return fig


def predict_with_probs(model, images):
    output = model(images)
    _, predictions_ = torch.max(output, 1)
    predictions = np.squeeze(predictions_.cpu().numpy())
    return predictions, [F.softmax(el, dim=0)[i].item() for i, el in zip(predictions, output)]


def plot_classes_preds(model, images, labels, classes):
    preds, probs = predict_with_probs(model, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure()
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        imshow(images[idx].cpu().detach())
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


def weight_histograms_conv2d(writer, step, weights, name):
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    for k in range(num_kernels):
        flattened_weights = weights[k].flatten()
        flattened_weights = flattened_weights[flattened_weights!=0]
        tag = f"{name}/kernel_{k}"
        writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')



def weight_histograms_linear(writer, step, weights, name):
    flattened_weights = weights.flatten()
    flattened_weights = flattened_weights[flattened_weights != 0]
    tag = name
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')
    # print('layer %s | std: %.3f | sparsity: %.3f%%' % (
    #    name, torch.std(flattened_weights), (flattened_weights == 0.).sum() / len(flattened_weights) * 100))


def weight_histograms(writer, step, model):
    # print("Visualizing model weights...")
    # Iterate over all model layers
    for name, module in model.named_modules():
        # Compute weight histograms for appropriate layer
        if isinstance(module, nn.Conv2d):
            weights = module.weight
            weight_histograms_conv2d(writer, step, weights, name)
        elif isinstance(module, nn.Linear):
            weights = module.weight
            weight_histograms_linear(writer, step, weights, name)
