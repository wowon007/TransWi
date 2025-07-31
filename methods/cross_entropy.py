import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from methods.utils.basemodel import BaseModel


class Cross_Entropy(BaseModel):
    NAME = 'cross_entropy'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, net, args, transform):
        super(Cross_Entropy, self).__init__(net, args, transform)
        self.opt_name = getattr(args, 'opt_name', 'adamw')

    def ini(self):
        if self.opt_name == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        elif self.opt_name == 'adamw':
            self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        elif self.opt_name == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.8, weight_decay=1e-5)
        else:
            raise ValueError(f"Unsupported optimizer: {self.opt_name}")
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                       T_max=int(self.local_epoch * self.eval_epoch),
                                                                       eta_min=1e-7)
        self.criterion = nn.CrossEntropyLoss()

    def loc_update(self, train_loader):
        self._train_net(train_loader)
        return None

    def _train_net(self, train_loader):
        net = self.net.to(self.device)
        net.train()

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            loss_list = []
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.data)
            iterator.desc = "Loss = %0.3f" % (sum(loss_list) / len(loss_list))
            # self.lr_scheduler.step()
            # print(self.lr_scheduler.get_lr())