from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import List
from torchvision import datasets
import numpy as np
import torch.optim


class BaseDataset:
    NAME = None
    SETTING = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> DataLoader:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass


def partition_dataset(datasets: list, batch_size: int, train: bool, merged_dataloader: bool = True) -> List[DataLoader]:
    ini_len_dict = {}
    not_used_index_dict = {}
    all_labels_list = []
    loaders = []
    train_samplers_idx = []
    test_samplers_idx = []
    train_start_pos = 0
    test_start_pos = 0
    default_status = train

    # get unique label idx
    for i in range(len(datasets)):
        name = datasets[i].data_name
        all_index = np.array(datasets[i].train_index_list)
        if name not in not_used_index_dict:
            not_used_index_dict[name] = all_index
            ini_len_dict[name] = len(all_index)

        all_labels_list.append(np.unique(np.array(datasets[i].imagefolder_obj.targets)[all_index]))

    all_labels_array = np.array(all_labels_list)

    # set sampler idx
    for index in range(len(datasets)):
        name = datasets[index].data_name

        dataset = datasets[index]

        idxs = np.random.permutation(datasets[index].train_index_list)
        selected_idx = idxs[:ini_len_dict[name]]

        train_labels = np.array(datasets[index].imagefolder_obj.targets)
        selected_labels = train_labels[selected_idx]

        show_up_num = np.zeros(len(all_labels_array[index]))
        for i in range(len(selected_labels)):
            label = selected_labels[i]
            show_up_num[label] += 1

        ## ensure minimum 1 sample for each label
        not_used_labels = np.where(show_up_num == 0)[0]
        for i in range(len(not_used_labels)):
            # select a sample to add
            not_used_label = not_used_labels[i]
            not_used_label_idx = np.where(train_labels == not_used_label)[0]
            add_index = not_used_label_idx[np.random.randint(len(not_used_label_idx))]

            # select a class to delete
            used_label = train_labels[selected_idx]
            prob_del_place = np.where(show_up_num >= 2)[0]
            del_index = np.random.randint(len(prob_del_place))
            del_label = prob_del_place[del_index]

            # select a sample to delete
            prob_del_selected = np.where(used_label == del_label)[0]
            del_index_selected = prob_del_selected[np.random.randint(len(prob_del_selected))]
            selected_idx = selected_idx[selected_idx != selected_idx[del_index_selected]]
            selected_idx = np.append(selected_idx, add_index)

            show_up_num[del_label] -= 1
            show_up_num[not_used_label] += 1

        # find the deleted idx in last step
        not_select_index = np.array(idxs)
        for i in range(len(selected_idx)):
            not_select_index = not_select_index[not_select_index != selected_idx[i]]

        not_used_index_dict[name] = not_select_index

        ## set sample idx
        train_samplers_idx += list(train_start_pos + np.array(selected_idx))
        ### get not select index
        len_ds = len(dataset.train_index_list) + len(dataset.test_index_list)
        not_select_idx = np.ones((len_ds,), dtype=bool)
        not_select_idx[dataset.train_index_list] = False
        not_select_idx = np.where(not_select_idx)[0]
        shuffle_idx = np.random.permutation(len(not_select_idx))
        not_select_idx = not_select_idx[shuffle_idx]
        test_samplers_idx += list(test_start_pos + np.array(not_select_idx))
        ## don't need to consider the test idx
        train_start_pos += len_ds
        test_start_pos += len_ds

        if not merged_dataloader:
            dataset.train_sampler = SubsetRandomSampler(selected_idx)
            dataset.test_sampler = SubsetRandomSampler(not_select_idx)
            dataset.train_index_list = train_samplers_idx
            dataset.test_index_list = not_select_idx
            loader = DataLoader(dataset,
                                batch_size=batch_size, sampler=dataset.train_sampler)
            if default_status != True:
                set_dataloader_status(loader, training_status=default_status)
            loaders.append(loader)


    if merged_dataloader:
        # shuffle in the datasets level
        train_samplers_idx = np.array(train_samplers_idx)
        idxs = np.random.permutation(len(train_samplers_idx))
        train_samplers_idx = train_samplers_idx[idxs]

        test_samplers_idx = np.array(test_samplers_idx)
        idxs = np.random.permutation(len(test_samplers_idx))
        test_samplers_idx = test_samplers_idx[idxs]

        # merge all datasets
        concat_dataset = ConcatDataset(datasets)
        concat_dataset.data_name = 'merged_dataset'

        concat_dataset.train_index_list = train_samplers_idx
        concat_dataset.test_index_list = test_samplers_idx

        # save sampler status
        concat_dataset.train_sampler = SubsetRandomSampler(train_samplers_idx)
        concat_dataset.test_sampler = SubsetRandomSampler(test_samplers_idx)

        concat_dataset.train = default_status

        loader = DataLoader(concat_dataset,
                            batch_size=batch_size, sampler=concat_dataset.train_sampler)
        if default_status != True:
            set_dataloader_status(loader, training_status=default_status)
        loaders.append(loader)
    return loaders


def random_loaders(train_dataset: datasets,
                   setting: BaseDataset) -> DataLoader:
    public_scale = setting.args.public_len
    y_train = train_dataset.targets
    n_train = len(y_train)
    idxs = np.random.permutation(n_train)
    if public_scale != None:
        idxs = idxs[0:public_scale]
    train_sampler = SubsetRandomSampler(idxs)
    train_loader = DataLoader(train_dataset, batch_size=setting.args.public_batch_size, sampler=train_sampler,
                              num_workers=4)
    setting.train_loader = train_loader

    return setting.train_loader


def set_dataloader_status(dataloader:DataLoader, training_status:bool):
    dataset = dataloader.dataset

    if hasattr(dataset, 'train') and dataset.train is training_status:
        return dataloader
    # set the dataset status
    dataset.train = training_status
    # load the sampler status
    if hasattr(dataloader, 'sampler'):
        if training_status is True:
            if hasattr(dataset, 'train_sampler'):
                dataloader = DataLoader(dataset, batch_size=dataloader.batch_size, sampler=dataset.train_sampler)
            else:
                exit('train_sampler is not set')
        else:
            if hasattr(dataset, 'test_sampler'):
                dataloader = DataLoader(dataset, batch_size=dataloader.batch_size, sampler=dataset.test_sampler)
            else:
                exit('test_sampler is not set')
    return dataloader

def get_dataloader_status(dataloader)->bool:
    dataset = dataloader.dataset
    if hasattr(dataset, 'train'):
        return dataset.train
    else:
        exit('get_dataset_status: dataset is not ConcatDataset or does not have train attribute')