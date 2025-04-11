'''
Project         : Rich Teacher Features for Efficient Single-Image Haze Removal
Lab             : Medical Image Computing and Artificial Intelligence Lab, National Institute of Technology, Trichy
Contributors    : Sai Mitheran, Anushri Suresh, Varun P. Gopi
'''

import torch

# It takes in two feature maps, downsamples them, flattens them, and then calculates the L2 norm of
# the difference between the two flattened feature maps's gram.


class FA_Module(torch.nn.Module):
    def __init__(self, subscale=0.25):
        """
        The function takes in a parameter called subscale. The function then creates a variable called
        HR_subscale, calculated from the ratio of the size of the high resolution image to the low resolution image

        :param subscale: the scale of the pooling operation
        """
        super(FA_Module, self).__init__()
        self.subscale = int(1 / subscale)
        self.HR_subscale = self.subscale * 2

    def forward(self, feature1, feature2):
        """
        :param feature1: the feature map of the HR image
        :param feature2: the feature map of the LR image
        :return: The L2 norm of the difference between the two matrices.
        """
        feature1 = torch.nn.AvgPool2d(self.HR_subscale)(feature1)
        feature2 = torch.nn.AvgPool2d(self.subscale)(feature2)

        (m_batchsize, C, height, width) = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width * height)
        mat1 = torch.bmm(feature1.permute(0, 2, 1), feature1)

        (m_batchsize, C, height, width) = feature2.size()
        feature2 = feature2.view(m_batchsize, -1, width * height)
        mat2 = torch.bmm(feature2.permute(0, 2, 1), feature2)

        L2norm = torch.norm(mat2 - mat1, 2)
        return L2norm / (height * width) ** 2
