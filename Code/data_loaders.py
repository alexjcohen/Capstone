import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import cv2
from PIL import Image
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
        img = transforms.functional.to_tensor(img).to(self.device)
        return {'id': self.imgids[item], 'img': img}


class AddNoise(object):
## IN PROGRESS - NEED TO UDPATE CLASS FOR NOISE ADDITION TO IMAGES
# TO DO:
#   - save np array of noise to be consistent across images
    def __init__(self, mean, std):

        self.mean = mean
        self.std = std

    def __call__(self, sample):
        np.random.seed(0)
        noise = np.random.normal(self.mean, self.std, (sample.shape[0], sample.shape[1]))
        sample_out = np.copy(sample)
        sample_out[:, :, 0] = sample_out[:, :, 0] + noise
        sample_out[:, :, 1] = sample_out[:, :, 1] + noise
        sample_out[:, :, 2] = sample_out[:, :, 2] + noise

        return sample_out

