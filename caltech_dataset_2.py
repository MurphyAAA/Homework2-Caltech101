from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import numpy as np
import pandas as pd


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def getIdxfromVal(arr, value):
    for i in range(len(arr)):
        if arr[i] == value:
            return i
    return -1


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt')

        split_path = './Caltech101/' + split + '.txt'
        img_path = './Caltech101/101_ObjectCategories/'
        with open(split_path) as f:
            # line = f.readline().strip('\n')
            line = f.read().splitlines()
            # print(line[0])
            self.data = np.array(line)  # 初始化
            imgs = np.array(line).astype(Image.Image)  # 只是为了初始化
            labels = np.zeros(self.data.shape).astype(np.str)  # 初始化 保存了小写字符串的label
        # print(len(self.data))

        for i in range(len(self.data)):  # 6096条数据
            # print(self.data[i])
            imgs[i] = pil_loader(img_path + self.data[i])  #所有的图
            labels[i] = self.data[i].split('/')[0].lower()  #所有的label （6096条）
            # print(self.data[i].split('/')[0])

        self.labels_unique = np.unique(labels) # 102条不重复的label
        self.data = list()
        # print(a.shape)
        for i in range(len(self.labels_unique)):
            img_cate = imgs[(labels == self.labels_unique[i])]
            self.data.append(img_cate)
            # print(i, '   ', a,'   ',len(a))
            # self.data = np.vstack((self.data,a))

        # self.data应该是102行若干列，行表示第几类，这一类的都在后面堆着
        # self.data = pd.DataFrame(self.data, columns=['image', 'label'])


        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        print(self.data)
        # print(
        image = self.data[index].astype(np.uint8)
        print(image)

        print(1)

        # image = self.data.iloc[index, 0].values.astype(np.uint8).reshape((3, 28, 28))
        # label = self.data.iloc[index, 1:]

        # Image should be a PIL Image
        # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image
        # return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data)  # Provide a way to get the length (number of elements) of the dataset
        return length
