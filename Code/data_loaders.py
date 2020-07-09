import torch
from torch.utils.data.dataset import Dataset
import os
import cv2
from PIL import Image, ImageFilter
from torchvision import transforms
import numpy as np
from pycocotools.coco import COCO


class AddNoise(object):
    """
    class to add noise to image by overlaying an n x m mask of randomly increased/decresed channel values
    :return: image with added noise
    """
    def __init__(self, mean, std, seed):

        self.mean = mean
        self.std = std
        self.seed = seed

    def __call__(self, sample):
        np.random.seed(self.seed)
        noise0 = np.random.normal(self.mean, self.std, (sample.size[1], sample.size[0]))
        noise1 = np.random.normal(self.mean, self.std, (sample.size[1], sample.size[0]))
        noise2 = np.random.normal(self.mean, self.std, (sample.size[1], sample.size[0]))
        sample_out = np.copy(sample)
        sample_out[:, :, 0] = sample_out[:, :, 0] + noise0 - 2*((noise0 + sample_out[:, :, 0] < 0) | (noise0 + sample_out[: , :, 0] > 255))
        sample_out[:, :, 1] = sample_out[:, :, 1] + noise1 - 2*((noise1 + sample_out[:, :, 1] < 0) | (noise1 + sample_out[: , :, 1] > 255))
        sample_out[:, :, 2] = sample_out[:, :, 2] + noise2 - 2*((noise2 + sample_out[:, :, 2] < 0) | (noise2 + sample_out[: , :, 2] > 255))

        return sample_out


class AddBlur(object):
    """
    class to add Gaussian blur to an image given an provided radius
    :return: blurred image
    """
    def __init__(self, radius):

        self.radius = radius

    def __call__(self, sample):

        sample_out = sample.filter(ImageFilter.GaussianBlur(radius=self.radius))

        return sample_out


class Rescale(object):
    def __call__(self, sample):
        sample_out = (sample-sample.min())/(sample.max() - sample.min())
        return sample_out


class ValImageLoader(Dataset):

    """
    Data loader for validation images
    :param filepath: path to image files
    :param annots_path: path to annotations json file
    :param device: device to run loader (cpu or cuda)
    :param transform: provided transforms for the images
    :return: image ID (for json file) and the image tensor
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


class TrainImageLoader(Dataset):

    """
    Data loader for training images
    :param filepath: path to image files
    :param annots_path: path to annotations json file
    :param device: device to run loader (cpu or cuda)
    :param transform: provided transforms for the images
    :return: image tensor and the target dictionary of bounding boxes and labels
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
        img = Image.open(img_name).convert('RGB')

        # get bounding boxes for image
        boxes = [a['bbox'] for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=pic, iscrowd=None))]
        for i in range(len(boxes)):
            boxes[i][2] = boxes[i][2] + boxes[i][0]
            boxes[i][3] = boxes[i][3] + boxes[i][1]

        boxes = torch.as_tensor(boxes, dtype=torch.float32).to(self.device)

        # # get labels
        labels = [a['category_id'] for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=pic, iscrowd=None))]
        labels = torch.LongTensor(np.array(labels)).to(self.device)

        target = dict()
        target['boxes'] = boxes
        target['labels'] = labels

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class TrainImageLoaderResize(Dataset):

    """
    Data loader for training images
    :param filepath: path to image files
    :param annots_path: path to annotations json file
    :param device: device to run loader (cpu or cuda)
    :param transform: provided transforms for the images
    :param size (tuple): desired size of output image (resizing input image)
    :return: image tensor and the target dictionary of bounding boxes and labels
    """

    def __init__(self, filepath, annots_path, device, size, transform=None):
        self.filepath = filepath
        self.transform = transform
        self.annots = annots_path
        self.coco = COCO(self.annots)
        self.imgids = self.coco.getImgIds()
        self.device = device
        self.size = size

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, item):
        # get image id
        pic = self.imgids[item]

        # get image name/filepath
        img_name = os.path.join(self.filepath, self.coco.loadImgs(pic)[0].get('file_name'))

        # load image, convert to RGB, and resize
        img_unscale = Image.open(img_name).convert('RGB')
        img = Image.open(img_name).convert('RGB').resize(self.size)

        width_ratio = img.width / img_unscale.width
        height_ratio = img.height / img_unscale.height

        # get bounding boxes for image
        boxes = [a['bbox'] for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=pic, iscrowd=None))]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).view(-1, 4).to(self.device)

        # # get labels
        labels = [a['category_id'] for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=pic, iscrowd=None))]
        labels = torch.LongTensor(np.array(labels)).view(-1).to(self.device)

        target = dict()
        target['boxes'] = boxes
        target['labels'] = labels
        target['w_r'] = width_ratio
        target['h_r'] = height_ratio

        if self.transform is not None:
            img = self.transform(img)

        return img, target
