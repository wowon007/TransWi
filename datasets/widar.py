import os

from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from datasets.utils.base_dataset import BaseDataset, partition_dataset

from utils.conf import data_path_widar
from torchvision.datasets import ImageFolder, DatasetFolder

from typing import List, Tuple
from PIL import Image

from backbone.TransWiNet import TransWiNet, TransWiNet_Small


class ImageFolderCustom(DatasetFolder):
    def __init__(self, data_name, root, train, transform=None, target_transform=None, subset_train_num=7,
                 subset_capacity=10):
        self.train = train
        self.data_name = data_name
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.imagefolder_obj = ImageFolder(self.root, self.transform,
                                           self.target_transform)

        all_data = self.imagefolder_obj.samples
        self.train_index_list = []
        self.test_index_list = []
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_index_list.append(i)
            else:
                self.test_index_list.append(i)

    def __len__(self):
        return len(self.imagefolder_obj.samples)


    def __getitem__(self, index):

        path = self.imagefolder_obj.samples[index][0]
        target = self.imagefolder_obj.samples[index][1]
        target = int(target)
        img = self.imagefolder_obj.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class Widar(BaseDataset):
    NAME = f'widar'

    def __init__(self, args):
        super(Widar, self).__init__(args)
        self.DATA_PATH = data_path_widar()
        self.subset_train_num = 8  # ratio of 10

        self.N_SAMPLES_PER_Class = None
        self.N_CLASS = 6
        if hasattr(args, 'DATA_SHAPE'):
            self.DATA_SHAPE = args.DATA_SHAPE
        else:
            self.DATA_SHAPE = (128, 128)

        self.DATA_OPERATE = [
            transforms.Resize(self.DATA_SHAPE),
            transforms.ToTensor(),
        ]
        if hasattr(args, 'transpose'):
            self.transpose = args.transpose
            self.DATA_OPERATE = [transforms.Lambda(
                lambda img: img.transpose(method=Image.TRANSPOSE))] + self.DATA_OPERATE

        self.Nor_TRANSFORM = transforms.Compose(
            self.DATA_OPERATE
        )

    def get_data_loaders(self) -> Tuple[List[DataLoader], List[DataLoader]]:
        nor_transform = self.Nor_TRANSFORM
        train_dataset_list = []
        test_transform = transforms.Compose(
            self.DATA_OPERATE
        )
        train_root_path = self.DATA_PATH + '/train'
        train_dirs = os.listdir(train_root_path)
        for _, domain in enumerate(train_dirs):
            train_dataset = ImageFolderCustom(data_name=domain, root=train_root_path + '/' + domain, train=True,
                                              transform=nor_transform, subset_train_num=self.subset_train_num,
                                              subset_capacity=10)
            train_dataset_list.append(train_dataset)

        train_loader = \
            partition_dataset(train_dataset_list, batch_size=self.args.batch_size, train=True, merged_dataloader=True)[
                0]

        train_loader.dataset.data_name = self.NAME

        test_root_path = self.DATA_PATH + '/test'
        test_dirs = os.listdir(test_root_path)
        test_loaders = []
        for _, domain in enumerate(test_dirs):
            # most of the test data is used for evaluation
            test_dataset = ImageFolderCustom(data_name=domain, root=test_root_path + '/' + domain, train=False,
                                             transform=test_transform, subset_train_num=2, subset_capacity=10)
            test_dataset.train_sampler = SubsetRandomSampler(test_dataset.train_index_list)
            test_dataset.test_sampler = SubsetRandomSampler(test_dataset.test_index_list)
            test_loaders.append(
                DataLoader(test_dataset, batch_size=self.args.batch_size, sampler=test_dataset.test_sampler))

        return [train_loader], test_loaders

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.Nor_TRANSFORM])
        return transform

    def get_backbone(self, net_name, args):
        nets_dict = {'transwi': TransWiNet, 'transwi-s': TransWiNet_Small}
        return nets_dict[net_name](num_classes=self.N_CLASS, args=args)
