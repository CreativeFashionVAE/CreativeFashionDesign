#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MyDataset(Dataset):
    """Dataloader class to load the images batch per batch
        ----------
        transform: defines wanted transformations for the images
        attribute_df: used for potential conditional vae; not used for this thesis
        type: specifies the type of data, namely, patterns, mvc etc.
        Used for current implementation are 'Patterns' and 'MVC_ST'

        Implementation References
        --------------
        Inspired by the official tutorial https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        Changed to fit the corresponding datasets
    """
    def __init__(self, root_dir, transform=None, attribute_df=None, type='MVC'):
        self.attributes_frame = attribute_df
        self.root_dir = root_dir
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.attributes_frame)

    def __getitem__(self, itemId):
        """Method that iterates over the csv-data in order to load images batch per batch.
            Customized to fit the different datasets. Method loads the data based on the
            filename attribute in the csv file. Attributes are not used as no conditional VAE is implemented.
        """

        if torch.is_tensor(itemId):
            itemId = itemId.tolist()

        if self.type == 'shapes':
            img_name = os.path.join(self.root_dir, str(self.attributes_frame['itemN'].iloc[itemId]) +"_"+str(self.attributes_frame['viewId'].iloc[itemId])+".jpg")
            attributes = self.attributes_frame.iloc[itemId, 11:]
        elif self.type == 'styles':
            img_name = os.path.join(self.root_dir, self.attributes_frame.iloc[itemId].loc['filename'])
            attributes = []


        image = io.imread(img_name)

        attributes = np.array(attributes, dtype=np.int64)
        sample = {'image': image, 'attributes': attributes}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Class that rescales the images based on the transformation.
    Each batch is resized when loaded by simple transform.resize.
    Simplified implementation as images are always resized to static values.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, attributes = sample['image'], sample['attributes']

        new_h, new_w = self.output_size, self.output_size

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'attributes': attributes}


class RandomCrop(object):
    """Crop Class taken from the tutorial and changed slightly.
    However, crop is not used at the moment, maybe needed for future work.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, attributes = sample['image'], sample['attributes']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'attributes': attributes}


class ToTensor(object):
    """Method that converts the numpy image to tensors and swaps the axis
    """

    def __call__(self, sample):
        image, attributes = sample['image'], sample['attributes']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'attributes': torch.from_numpy(attributes)}
    
"""Currently used fashion dataset that consists of al Shirts&Tops images"""
def create_new_data_squared(img_size=256):
    df_att = pd.read_csv('./data/preprocessed_shapes_metadata.csv')
    return MyDataset(root_dir='./data/preprocessed_data_shapes/',
                     type = 'shapes',
                     attribute_df=df_att,
                     transform=transforms.Compose([
                         Rescale(img_size),
                         ToTensor()
                     ]))

    
"""Currently used patterns dataset"""
def create_newPatt_dataset(img_size=128):
    df_att = pd.read_csv('./data/style_metadata.csv', delimiter=';')
    print(df_att.head(5))
    return MyDataset(root_dir='./data/preprocessed_data_styles/',
                     type = 'styles',
                     attribute_df=df_att,
                     transform=transforms.Compose([
                         Rescale(img_size),
                         ToTensor()
                     ]))

