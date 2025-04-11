'''
Project         : Rich Teacher Features for Efficient Single-Image Haze Removal
Lab             : Medical Image Computing and Artificial Intelligence Lab, National Institute of Technology, Trichy
Contributors    : Sai Mitheran, Anushri Suresh, Varun P. Gopi
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import RESIDE_Dataset
import numpy as np
from fa import FA_Module
from utils import *
from full_model import KD_Net


def train_model(
    model,
    criterion,
    optimizer,
    dataload,
    num_epochs,
    device="cuda:0",
    w_fa=0.25,
    its_test_path="..\\dataset\\SOTS\\indoor",
    resume_epoch=None,
):

    """
    It trains the model, saves the model weights, saves the model evaluation metrics, and saves the
    model itself.

    :param model: the model to train
    :param criterion: The loss function we're using
    :param optimizer: The optimizer used to train the model
    :param dataload: the dataloader for the training data
    :param num_epochs: Number of epochs to train the model
    :param device: the device to run the training on, defaults to cuda:0 (optional)
    :param w_fa: weight for feature alignment loss
    :param its_test_path: the path to the test dataset, defaults to ..\dataset\SOTS\indoor (optional)
    :param resume_epoch: The epoch to resume training from. If None, training starts from scratch
    :return: The model is being returned.
    """

    losses = []
    start_step = 0
    max_ssim = max_psnr = 0
    ssims, psnrs = [], []
    min_loss = 100
    FA_module = FA_Module().to(device)

    for epoch in range(resume_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for haze, clear_LR, clear in dataload:
            step += 1
            haze = haze.to(device)
            clear_LR = clear_LR.to(device)
            clear = clear.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            guiding_teacher_features, outputs = model(haze, clear_LR)

            # Loss functions
            loss_fa = FA_module(guiding_teacher_features, outputs)
            loss_dehaze = criterion(outputs, clear_LR)
            loss = loss_dehaze + w_fa * loss_fa

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.student.parameters(), 0.1)
            optimizer.step()
            epoch_loss += loss.item()

        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))

        if epoch_loss / step <= min_loss:
            min_loss = epoch_loss / step
            torch.save(
                model.student.state_dict(), "..\\new_modified_network_weights_HR.pth"
            )
            print("saved")

            loader_test = DataLoader(
                dataset=RESIDE_Dataset(its_test_path, train=False, size="whole img"),
                batch_size=1,
                shuffle=False,
            )

            with torch.no_grad():
                ssim_eval, psnr_eval = test(
                    model, loader_test, max_psnr, max_ssim, step
                )

            print(f"\nstep: {step} | ssim: {ssim_eval:.4f} | psnr: {psnr_eval:.4f}")

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "max_psnr": max_psnr,
                        "max_ssim": max_ssim,
                        "ssims": ssims,
                        "psnrs": psnrs,
                        "losses": losses,
                        "model": model.student.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f"..\\eval_HR_{str(epoch)}.pth",
                )

                print(
                    f"\n model saved at step : {step} | max_psnr: {max_psnr:.4f} | max_ssim: {max_ssim:.4f}"
                )

    return model


def test(net, loader_test, max_psnr, max_ssim, step, device="cuda:0"):
    """
    It takes in a network, a test loader, the current best PSNR and SSIM, the current step, and the
    device to use. It then iterates through the test loader, and for each batch, it calculates the SSIM
    and PSNR between the predicted image and the ground truth image. It then returns the mean SSIM and
    PSNR

    :param net: the network
    :param loader_test: the test dataset
    :param max_psnr: the highest psnr value achieved so far
    :param max_ssim: the highest SSIM value achieved so far
    :param step: the current step of training
    :param device: the device to run the training on, defaults to cuda:0 (optional)
    :return: The mean SSIM and PSNR values for the test set.
    """
    net.student.eval()
    torch.cuda.empty_cache()
    ssims, psnrs = [], []
    for i, (haze, clear_LR, _) in enumerate(loader_test):
        haze = haze.to(device)
        clear_LR = clear_LR.to(device)
        pred = net.student(haze)

        ssim1 = ssim(pred, clear_LR).item()
        psnr1 = psnr(pred, clear_LR)
        ssims.append(ssim1)
        psnrs.append(psnr1)

    return np.mean(ssims), np.mean(psnrs)


def train(batch_size, num_epochs, resume=False, device="cuda:0"):
    """
    It trains the model for the number of epochs specified, and saves the best model (based on the
    lowest validation loss) in the `checkpoints` folder

    :param batch_size: The number of images to be used in each batch
    :param num_epochs: number of epochs to train for
    :param resume: if you want to resume training from a previous checkpoint, set this to True, defaults
    to False (optional)
    :param device: the device to run the training on, defaults to cuda:0 (optional)
    """
    its_train_path = "..\\dataset\\ITS"
    its_test_path = "..\\dataset\\SOTS\\indoor"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crop = True
    crop_size = "whole_img"
    if crop:
        crop_size = crop_size

    model = KD_Net().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=filter(lambda x: x.requires_grad, model.parameters()),
        lr=0.0001,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    train_iter = DataLoader(
        dataset=RESIDE_Dataset(its_train_path, train=True, size=crop_size),
        batch_size=batch_size,
        shuffle=True,
    )

    if resume:
        checkpoint_fpath = (
            "eval_16.pth"  # replace the path of the previous best epoch here
        )
        model.student, optimizer, resume_epoch = load_resume(
            checkpoint_fpath, model.student, optimizer
        )

    else:
        resume_epoch = 0

    train_model(
        model, criterion, optimizer, train_iter, num_epochs, resume_epoch=resume_epoch
    )


if __name__ == "__main__":
    train(batch_size=2, num_epochs=200, resume=False)
