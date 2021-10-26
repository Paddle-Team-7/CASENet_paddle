import os
import numpy as np
import time
import PIL
from PIL import Image

#import torch
import paddle
import paddle.nn as nn
#import torch.utils.data
#import torchvision.transforms as transforms
import paddle.vision.transforms as transforms
#from Paddle_ToPILImage import ToPILImage
#import torchvision.datasets as datasets

##--------------------------------------------------##
import numbers
from PIL import Image
from paddle.vision.transforms import BaseTransform
from paddle.vision.transforms import functional as F

class ToTensor(BaseTransform):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to ``numpy.ndarray`` with shapr (C x H x W).
    Args:
        data_format (str, optional): Data format of output tensor, should be 'HWC' or
            'CHW'. Default: 'CHW'.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.
    """

    def __init__(self, data_format='CHW', keys=None):
        super(ToTensor, self).__init__(keys)
        self.data_format = data_format

    def _apply_image(self, img):
        """
        Args:
            img (PIL.Image|np.ndarray): Image to be converted to tensor.
        Returns:
            np.ndarray: Converted image.
        """
        if isinstance(img, PIL.JpegImagePlugin.JpegImageFile) or isinstance(
                img, PIL.Image.Image):
            img = np.array(img)
        img = img / 255.0
        img = img.transpose((2, 0, 1)).astype("float32")
        img = paddle.to_tensor(img)
        return img

class ToPILImage(BaseTransform):
    def __init__(self, mode=None, data_format='CHW',keys=None):
        super(ToPILImage, self).__init__(keys)

    def _apply_image(self, pic):
        """
        Args:
            pic (Tensor|np.ndarray): Image to be converted to PIL Image.
        Returns:
            PIL: Converted image.
        """
        if not (isinstance(pic, paddle.Tensor) or isinstance(pic, np.ndarray)):
            raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(
                type(pic)))

        elif isinstance(pic, paddle.Tensor):
            if pic.ndimension() not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndimension()))

            elif pic.ndimension() == 2:
                # if 2D image, add channel dimension (CHW)
                pic = pic.unsqueeze(0)

        elif isinstance(pic, np.ndarray):
            if pic.ndim not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndim))

            elif pic.ndim == 2:
                # if 2D image, add channel dimension (HWC)
                pic = np.expand_dims(pic, 2)

        npimg = pic
        if isinstance(pic, paddle.Tensor) and "float" in str(pic.numpy(
        ).dtype) and mode != 'F':
            pic = pic.mul(255).byte()
        if isinstance(pic, paddle.Tensor):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))

        if not isinstance(npimg, np.ndarray):
            raise TypeError(
                'Input pic must be a paddle.Tensor or NumPy ndarray, ' +
                'not {}'.format(type(npimg)))

        if npimg.shape[2] == 1:
            expected_mode = None
            npimg = npimg[:, :, 0]
            if npimg.dtype == np.uint8:
                expected_mode = 'L'
            elif npimg.dtype == np.int16:
                expected_mode = 'I;16'
            elif npimg.dtype == np.int32:
                expected_mode = 'I'
            elif npimg.dtype == np.float32:
                expected_mode = 'F'
            if mode is not None and mode != expected_mode:
                raise ValueError(
                    "Incorrect mode ({}) supplied for input type {}. Should be {}"
                    .format(mode, np.dtype, expected_mode))
            mode = expected_mode

        elif npimg.shape[2] == 2:
            permitted_2_channel_modes = ['LA']
            if mode is not None and mode not in permitted_2_channel_modes:
                raise ValueError("Only modes {} are supported for 2D inputs".
                                 format(permitted_2_channel_modes))

            if mode is None and npimg.dtype == np.uint8:
                mode = 'LA'

        elif npimg.shape[2] == 4:
            permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
            if mode is not None and mode not in permitted_4_channel_modes:
                raise ValueError("Only modes {} are supported for 4D inputs".
                                 format(permitted_4_channel_modes))

            if mode is None and npimg.dtype == np.uint8:
                mode = 'RGBA'
        else:
            permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
            if mode is not None and mode not in permitted_3_channel_modes:
                raise ValueError("Only modes {} are supported for 3D inputs".
                                 format(permitted_3_channel_modes))
            if mode is None and npimg.dtype == np.uint8:
                mode = 'RGB'

        if mode is None:
            raise TypeError('Input type {} is not supported'.format(
                npimg.dtype))

        return Image.fromarray(npimg, mode=mode)

##----------------------------------------------------##
import sys
sys.path.append("../")

from dataloader.cityscapes_data import CityscapesData

import config

class RGB2BGR(object):
    """
    Since we use pretrained model from Caffe, need to be consistent with Caffe model.
    Transform RGB to BGR.
    """
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img):
        if img.mode == 'L':
            return np.concatenate([np.expand_dims(img, 2)], axis=2) 
        elif img.mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(img)[:, :, ::-1]], axis=2)
            else:
                return np.concatenate([np.array(img)], axis=2)

class ToTorchFormatTensor(object):
    """
    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] or [0, 255]. 
    """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = paddle.to_tensor(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            #img = paddle.toTensor(torch.ByteStorage.from_buffer(pic.tobytes()),dtype='uint8')
            img = paddle.toTensor(pic.tobytes(),dtype='uint8')
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        
        return img.float().div(255) if self.div else img.float()

def get_dataloader(args):
    # Define data files path.
    root_img_folder = "/ais/gobi4/fashion/edge_detection/data_aug" 
    root_label_folder = "/ais/gobi4/fashion/edge_detection/data_aug"
    train_anno_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_train_aug.txt"
    val_anno_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_test.txt"
    train_hdf5_file = "/ais/gobi6/jiaman/github/CASENet/utils/train_aug_label_binary_np.h5"
    val_hdf5_file = "/ais/gobi6/jiaman/github/CASENet/utils/test_label_binary_np.h5"

    input_size = 472
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])

    train_augmentation = transforms.Compose([transforms.RandomResizedCrop(input_size, scale=(0.75,1.0), ratio=(0.75,1.0)), transforms.RandomHorizontalFlip()])
    train_label_augmentation = transforms.Compose([transforms.RandomResizedCrop(input_size, scale=(0.75,1.0), ratio=(0.75,1.0), interpolation=PIL.Image.NEAREST), \
                                transforms.RandomHorizontalFlip()])

    train_dataset = CityscapesData(
        root_img_folder,
        root_label_folder,
        train_anno_txt,
        train_hdf5_file,
        input_size,
        cls_num=args.cls_num,
        img_transform = transforms.Compose([
                        train_augmentation,
                        RGB2BGR(roll=True),
                        ToTorchFormatTensor(div=False),
                        normalize,
                        ]),
        label_transform = transforms.Compose([
                        ToPILImage(),
                        train_label_augmentation,
                        transforms.ToTensor(),
                        ]))
    train_loader = paddle.io.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)#pinmemory删除
    
    val_dataset = CityscapesData(
        root_img_folder,
        root_label_folder,
        val_anno_txt,
        val_hdf5_file,
        input_size,
        cls_num=args.cls_num,
        img_transform = transforms.Compose([
                        transforms.Resize([input_size, input_size]),
                        RGB2BGR(roll=True),
                        ToTorchFormatTensor(div=False),
                        normalize,
                        ]),
        label_transform = transforms.Compose([
                        ToPILImage(),
                        transforms.Resize([input_size, input_size], interpolation=PIL.Image.NEAREST),
                        transforms.ToTensor(),
                        ]))
    val_loader = paddle.io.DataLoader(
        val_dataset, batch_size=args.batch_size/2, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    return train_loader, val_loader

if __name__ == "__main__":
    args = config.get_args()
    args.batch_size = 1
    train_loader, val_loader = get_dataloader(args)
    for i, (img, target) in enumerate(val_loader):
        print("target.size():{0}".format(target.size()))
        print("target:{0}".format(target))
        break;
