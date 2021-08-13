import os.path
import random

import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
from torchvision.datasets.folder import find_classes
from torchvision.datasets.folder import make_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch

import util.utils as utils
from torchvision.transforms import functional as F
import copy


class AlignedConcDataset:

    def __init__(self, cfg, data_dir=None, transform=None):
        self.cfg = cfg
        self.transform = transform
        self.data_dir = data_dir

        self.classes, self.class_to_idx = find_classes(self.data_dir)
        if self.cfg.FILTER_BEDROOM: 
            # put the bedroom class at the end of class_to_idx in order to make make_dataset working
            # and, at the same, time not assigning high label int to other images
            excludedClass = "bedroom"
            self.classes = [cls for cls in self.classes if cls != excludedClass] #removing bedroom from self.classes
            self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
            self.class_to_idx[excludedClass] = None 
                            
            self.imgs = make_dataset(self.data_dir, self.class_to_idx, ['jpg','png']) #read all the classes

            if self.cfg.MIXED_SOURCE and data_dir != self.cfg.DATA_DIR_VAL :    #if it is mixed_source and 
                                                                                #we are not loading target domain
                tmp_secondSource_imgs = make_dataset(self.cfg.DATA_DIR_TRAIN_2, self.class_to_idx, ['jpg','png'])
                self.imgs.extend(tmp_secondSource_imgs) #append operation

            self.imgs = [(path,cls) for (path,cls) in self.imgs if cls != None] #removing images with class bedroom
            self.class_to_idx.pop(excludedClass)
            self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        else:
            self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
            self.imgs = make_dataset(self.data_dir, self.class_to_idx, ['jpg','png'])

        print(f"ADDED ------- len(self.imgs) : ",len(self.imgs))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        img_path, label = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc = Image.open(img_path).convert('RGB')

        # split RGB and Depth as A and B
        w, h = AB_conc.size
        w2 = int(w / 2)
        if w2 > self.cfg.FINE_SIZE:
            A = AB_conc.crop((0, 0, w2, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
            B = AB_conc.crop((w2, 0, w, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
        else:
            A = AB_conc.crop((0, 0, w2, h))
            B = AB_conc.crop((w2, 0, w, h))

        sample = {'A': A, 'B': B, 'img_name': img_name, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']

        if self.padding > 0:
            A = F.pad(A, self.padding)
            B = F.pad(B, self.padding)

        # pad the width if needed
        if self.pad_if_needed and A.size[0] < self.size[1]:
            A = F.pad(A, (int((1 + self.size[1] - A.size[0]) / 2), 0))
            B = F.pad(B, (int((1 + self.size[1] - B.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and A.size[1] < self.size[0]:
            A = F.pad(A, (0, int((1 + self.size[0] - A.size[1]) / 2)))
            B = F.pad(B, (0, int((1 + self.size[0] - B.size[1]) / 2)))

        i, j, h, w = self.get_params(A, self.size)
        sample['A'] = F.crop(A, i, j, h, w)
        sample['B'] = F.crop(B, i, j, h, w)

        return sample


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        sample['A'] = F.center_crop(A, self.size)
        sample['B'] = F.center_crop(B, self.size)
        return sample


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        if random.random() > 0.5:
            A = F.hflip(A)
            B = F.hflip(B)

        sample['A'] = A
        sample['B'] = B

        return sample


class Resize(transforms.Resize):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        h = self.size[0]
        w = self.size[1]

        sample['A'] = F.resize(A, (h, w))
        sample['B'] = F.resize(B, (h, w))

        return sample


class ToTensor(object):
    def __call__(self, sample):

        A, B = sample['A'], sample['B']

        sample['A'] = F.to_tensor(A)
        sample['B'] = F.to_tensor(B)

        return sample

class Normalize(transforms.Normalize):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        sample['A'] = F.normalize(A, self.mean, self.std)
        sample['B'] = F.normalize(B, self.mean, self.std)

        return sample

class Lambda(transforms.Lambda):

    def __call__(self, sample):
        return self.lambd(sample)