'''
Project         : Rich Teacher Features for Efficient Single-Image Haze Removal
Lab             : Medical Image Computing and Artificial Intelligence Lab, National Institute of Technology, Trichy
Contributors    : Sai Mitheran, Anushri Suresh, Varun P. Gopi
'''

import os
import torch
from torch.autograd import Variable
from math import exp
import numpy as np
import torch.nn.functional as F
from torchvision.utils import make_grid
import math
import matplotlib.pyplot as plt


def save_model(model, model_dir, model_filename):
    """
    > Save the model's state dictionary to a file in the specified directory

    :param model: the model to save
    :param model_dir: The directory where the model will be saved
    :param model_filename: The name of the file to save the model to
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):
    """
    > Loads a model from a filepath and returns the model

    :param model: the model to load
    :param model_filepath: The path to the saved model
    :param device: The device to use for training
    :return: The model is being returned.
    """
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model


def freeze_model(model):
    """
    > It takes a model as input and returns the same model with all of its parameters set to
    `requires_grad = False`

    :param model: The model to be trained
    :return: The model with the parameters frozen.
    """
    for param in model.parameters():
        param.requires_grad = False

    return model


# Metrics


def gaussian(window_size, sigma):
    """
    It creates a 1D Gaussian kernel with a given window size and standard deviation

    :param window_size: The size of the window to be used for the Gaussian blur
    :param sigma: the standard deviation of the Gaussian distribution
    :return: A 1D tensor of size window_size, where each element is the value of a gaussian function at
    that point.
    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    It creates a 2D gaussian kernel (to
    be used as a weight matrix) of the specified size for each channel in the input

    :param window_size: The size of the window to be used for the gaussian blur
    :param channel: number of channels in the image
    :return: A 2D gaussian window of size window_size x window_size
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    It takes two images, convolves them with a Gaussian kernel, and then calculates the SSIM between the
    two images

    :param img1: The first image batch
    :param img2: the second image
    :param window: The window function to use
    :param window_size: The size of the sliding window
    :param channel: the number of channels in the image
    :param size_average: If True, the output is averaged over all images in the batch, defaults to True
    (optional)
    :return: The mean of the ssim_map.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    """
    It takes two images, and returns a single number

    :param img1: The first image batch
    :param img2: The second image to compare to the first
    :param window_size: The side length of the sliding window used in comparison. Must be an odd value,
    defaults to 11 (optional)
    :param size_average: if True, the output is averaged over all images in the batch, defaults to True
    (optional)
    :return: The SSIM value
    """
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(pred, gt):
    """
    > The PSNR is the ratio between the maximum possible power of a signal and the power of corrupting
    noise that affects the fidelity of its representation

    :param pred: the predicted image
    :param gt: ground truth image
    :return: The PSNR value
    """
    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)


def tensorShow(tensors, titles=None):
    """
    It takes a list of tensors, and plots them in a grid

    :param tensors: a list of tensors to be displayed
    :param titles: The titles of the images
    """
    """t:BCWH"""
    fig = plt.figure()
    for tensor, title, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(211 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(title)
    plt.show()


def lr_schedule_cosdecay(t, T, init_lr=0.0001):
    """
    > The learning rate is a cosine function of the epoch number

    :param t: current epoch
    :param T: total number of iterations
    :param init_lr: the initial learning rate
    :return: The learning rate is being returned.
    """
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


# It takes in a VGG model and returns a loss function that computes the mean squared error between the
# features of the dehazed image and the ground truth image


class PerLoss(torch.nn.Module):
    def __init__(self, vgg_model):
        """
        It takes in a VGG model and creates a new model that outputs the feature maps of the VGG model at
        the layers we want

        :param vgg_model: The VGG model that we will use to extract features from the content and style
        images
        """
        super(PerLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {"3": "relu1_2", "8": "relu2_2", "15": "relu3_3"}

    def output_features(self, x):
        """
        It takes in an image, and returns the feature maps of the image at the layers we want

        :param x: the input image
        :return: The output is a list of the features of the image at each layer.
        """
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        """
        We are going to take the output of the model (the dehazed image) and the ground truth image and pass
        them through the VGG19 model. We will then take the output of the VGG19 model and calculate the MSE
        loss between the two

        :param dehaze: the output of the dehazing network
        :param gt: ground truth image
        :return: The loss is being returned.
        """
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss) / len(loss)


def load_resume(checkpoint_fpath, model, optimizer):
    """
    It loads the model and optimizer state dicts from the checkpoint file.

    :param checkpoint_fpath: the path to the checkpoint file
    :param model: the model you want to load
    :param optimizer: the optimizer to use for training
    :return: The model, optimizer, and the epoch number.
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"]
