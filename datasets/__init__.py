import os
import inspect
import importlib
from datasets.utils.base_dataset import BaseDataset
from argparse import Namespace

def get_all_models():
    return [model.split('.')[0] for model in os.listdir('datasets')
            if not model.find('__') > -1 and 'py' in model]

DS_NAMES = {}
for model in get_all_models():
    mod = importlib.import_module('datasets.' + model)
    dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'BaseDataset' in str(inspect.getmro(getattr(mod, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        DS_NAMES[c.NAME] = c



def get_dataset(args: Namespace) -> BaseDataset:

    assert args.dataset in DS_NAMES.keys()
    return DS_NAMES[args.dataset](args)

