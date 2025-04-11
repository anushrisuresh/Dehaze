'''
Project         : Rich Teacher Features for Efficient Single-Image Haze Removal
Lab             : Medical Image Computing and Artificial Intelligence Lab, National Institute of Technology, Trichy
Contributors    : Sai Mitheran, Anushri Suresh, Varun P. Gopi
'''

import torch
from torch import nn
from student import Student
from teacher import Teacher
from utils import *

# The KD_Net class takes in a low resolution hazy image and a high resolution clear image as inputs
# and outputs the dehazed image


class KD_Net(nn.Module):
    def __init__(self, device="cuda:0"):
        """
        We create three teacher models, one for each channel, and load the pre-trained weights for each of
        them. We then create a student model and pass the number of groups and blocks as parameters

        :param device: The device to run the model on, defaults to cuda:0 (optional)
        """
        super(KD_Net, self).__init__()

        self.device = device

        # Teacher Module
        teacher_R = Teacher(scale_factor=2).to(device)
        teacher_G = Teacher(scale_factor=2).to(device)
        teacher_B = Teacher(scale_factor=2).to(device)

        self.teacher_R = freeze_model(
            load_model(teacher_R, "checkpoints/DSCFSRCNN_HR_best_red.pth", device)
        ).eval()
        self.teacher_G = freeze_model(
            load_model(teacher_G, "checkpoints/DSCFSRCNN_HR_best_green.pth", device)
        ).eval()
        self.teacher_B = freeze_model(
            load_model(teacher_B, "checkpoints/DSCFSRCNN_HR_best_blue.pth", device)
        ).eval()

        # Student Module - To be trained
        self.student = Student(gps=3, blocks=6).to(device)

    def forward(self, haze, clear_LR):
        """
        The function takes in a hazy image and a clear image, and returns the guiding teacher features and
        the dehazed image

        :param haze: the hazy image
        :param clear_LR: the low resolution clear image
        :return: The guiding teacher features and the dehazed HR image
        """

        red_inputs = clear_LR[:, 0, :, :].unsqueeze(1)
        green_inputs = clear_LR[:, 1, :, :].unsqueeze(1)
        blue_inputs = clear_LR[:, 2, :, :].unsqueeze(1)

        # Do not train the teacher
        with torch.no_grad():
            # Teacher forward pass
            _, _, red_inputs = self.teacher_R(red_inputs)
            _, _, green_inputs = self.teacher_G(green_inputs)
            _, _, blue_inputs = self.teacher_B(blue_inputs)
            guiding_teacher_features = torch.cat(
                [red_inputs, green_inputs, blue_inputs], dim=1
            )  # clear_HR

        # Student forward pass
        dehazed_HR = self.student(haze)

        return guiding_teacher_features, dehazed_HR
