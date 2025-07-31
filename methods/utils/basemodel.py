import numpy as np
import torch.nn as nn
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device


class BaseModel(nn.Module):
    NAME = None
    N_CLASS = None

    def __init__(self, net: nn.Module,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(BaseModel, self).__init__()
        self.net = net
        self.args = args
        self.transform = transform

        # For Online
        self.random_state = np.random.RandomState()

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)
        self.dtype = torch.float32

        self.eval_epoch = args.eval_epoch
        self.local_epoch = args.local_epoch
        self.lr = args.lr
        self.trainloaders = None
        self.testlodaers = None

        self.epoch_index = 0  # Save the Communication Index

        self.net_to_device()

    def net_to_device(self):
        self.net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass

    def loc_update(self, priloader_list):
        pass
