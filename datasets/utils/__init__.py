import os
import inspect
import importlib
from datasets.utils.base_dataset import BaseDataset
from argparse import Namespace


def get_all_methods():
    return [method.split('.')[0] for method in os.listdir('datasets')
            if not method.find('__') > -1 and 'py' in method]


NAMES = {}
for method in get_all_methods():
    mod = importlib.import_module('datasets.' + method)
    dataset_classes_name = [x for x in mod.__dir__() if
                            'type' in str(type(getattr(mod, x))) and 'ContinualDataset' in str(
                                inspect.getmro(getattr(mod, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        NAMES[c.NAME] = c

    gcl_dataset_classes_name = [x for x in mod.__dir__() if
                                'type' in str(type(getattr(mod, x))) and 'GCLDataset' in str(
                                    inspect.getmro(getattr(mod, x))[1:])]
    for d in gcl_dataset_classes_name:
        c = getattr(mod, d)
        NAMES[c.NAME] = c


def get_dataset(args: Namespace) -> BaseDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
