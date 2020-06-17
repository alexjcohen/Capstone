import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import cv2
from PIL import Image, ImageFilter
from torchvision import transforms
import numpy as np
from pycocotools.coco import COCO


class ValImageLoader(Dataset):

    """
    Data loader for validation images
     returns image ID (for json file) and the image tensor (for now, to adjust w/ additional transforms)
    """

    def __init__(self, filepath, annots_path, device, transform=None):
        self.filepath = filepath
        self.transform = transform
        self.annots = annots_path
        self.coco = COCO(self.annots)
        self.imgids = self.coco.getImgIds()
        self.device = device

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, item):
        pic = self.imgids[item]
        img_name = os.path.join(self.filepath, self.coco.loadImgs(pic)[0].get('file_name'))
        img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        img = transforms.functional.to_tensor(img).to(self.device)
        return {'id': self.imgids[item], 'img': img}


class AddNoise(object):

    def __init__(self, mean, std, seed):

        self.mean = mean
        self.std = std
        self.seed = seed

    def __call__(self, sample):
        np.random.seed(self.seed)
        noise = np.random.normal(self.mean, self.std, (sample.size[1], sample.size[0]))
        sample_out = np.copy(sample)
        sample_out[:, :, 0] = sample_out[:, :, 0] + noise
        sample_out[:, :, 1] = sample_out[:, :, 1] + noise
        sample_out[:, :, 2] = sample_out[:, :, 2] + noise

        return sample_out


class AddBlur(object):

    def __init__(self, radius):

        self.radius = radius

    def __call__(self, sample):

        sample_out = sample.filter(ImageFilter.GaussianBlur(radius=self.radius))

        return sample_out
