import torch
from torch.utils.data import Dataset

import os
from os import walk
import random

from PIL import Image
import numpy as np

class SegmentationImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        _, _, self.filenames = next(walk(image_dir))

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        
    # @classmethod
    # def preprocess(cls, pil_img, normalize=True):
    #     pil_img = pil_img.convert('L')

    #     pil_img = pil_img.resize(IMAGE_SIZE)
    #     img_nd = np.array(pil_img)

    #     if len(img_nd.shape) == 2:
    #         img_nd = np.expand_dims(img_nd, axis=0)

    #     if normalize:
    #         img_trans = img_nd / 255

    #     return img_trans

    def __getitem__(self, idx):
        image = Image.open(self.image_dir + '/' + self.filenames[idx])
        mask  = Image.open(self.mask_dir  + '/' + self.filenames[idx])
        
        if self.transform:
        
            seed = random.randint(0, 2**32 - 1)

            image = self.transform[0](image)

            self._set_seed(seed)

            image = self.transform[1](image)

            self._set_seed(seed)

            mask  = self.transform[1](mask)
            
            if type(mask) is torch.Tensor:
                mask = torch.where(mask > 0.5, 1, 0)

        image = image.type(torch.FloatTensor)
        mask  =  mask.type(torch.FloatTensor)


        # image = self.preprocess(image)
        # mask = self.preprocess(mask)

        # image = torch.from_numpy(image).type(torch.FloatTensor)
        # mask = torch.from_numpy(mask).type(torch.FloatTensor)

        return image, mask

    def __len__(self):
        return len(self.filenames)
