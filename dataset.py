'''
Project         : Rich Teacher Features for Efficient Single-Image Haze Removal
Lab             : Medical Image Computing and Artificial Intelligence Lab, National Institute of Technology, Trichy
Contributors    : Sai Mitheran, Anushri Suresh, Varun P. Gopi
'''

import torch.utils.data as data
import torchvision.transforms as tfs
import os, random
from PIL import Image
from torchvision.transforms import functional as FF


# It takes in a path to the dataset, a boolean value to indicate whether it's training or not, and a
# size to resize the images to.
#
# It then loads the hazy images and the clear images, and returns a random crop of the images. If it's training, it also applies random horizontal flips and rotations.


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size="whole_img", format=".png"):
        """
        The function takes in the path to the dataset, a boolean value to indicate whether the dataset is
        for training or testing, the size of the image to be returned, and the format of the image. It then
        creates a list of all the hazy images in the dataset, and the path to the clear images. It also
        creates a resizer object to resize the images to the desired size.

        :param path: the path to the dataset
        :param train: whether the dataset is for training or testing
        :param size: the size of the image to be returned, defaults to whole_img (optional)
        :param format: the format of the images, defaults to .png (optional)
        """
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, "hazy"))
        self.haze_imgs = [os.path.join(path, "hazy", img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, "clear")
        self.resizer_2x = tfs.Resize((920, 1240))

    def __getitem__(self, index):
        """
        The function takes in the index of the image, and returns the haze image, the low resolution clear
        image, and the high resolution clear image

        :param index: the index of the image in the dataset
        :return: haze, clear_LR, clear
        """
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        id = img.split("\\")[-1].split("_")[0]
        clear_name = id + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(
                haze, output_size=(self.size, self.size)
            )
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        haze, clear_LR, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear_LR, clear

    def augData(self, data, target):
        """
        It takes in an image and a target, and returns the image, the target, and the target resized to half
        its size

        :param data: the path to the data folder
        :param target: the target image
        :return: data, target, target_lr_2x
        """
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        target_lr_2x = self.resizer_2x(target)

        return data, target, target_lr_2x

    def __len__(self):
        """
        The function returns the length of the dataset
        :return: The length of the haze_imgs list.
        """
        return len(self.haze_imgs)
