import time
import torch
from argparse import Namespace
from methods.utils.basemodel import BaseModel
from datasets.utils.base_dataset import BaseDataset, set_dataloader_status, get_dataloader_status
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from typing import List


def reverseDB2val(x) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return 10 ** (x / 20)


def global_evaluate(model: BaseModel, test_dls: List[DataLoader]) -> Tuple[
    list, list, list, list]:
    accs = []
    net = model.net
    status = net.training
    dtype = next(net.parameters())[0].dtype
    net.eval()
    ds_names = []
    all_features = []
    all_labels = []
    show_count = 500
    for j, test_dl in enumerate(test_dls):
        total, top1 = 0.0, 0.0
        tmp_features = []
        tmp_labels = []
        ## save the original status
        ds_status = get_dataloader_status(test_dl)
        test_dl = set_dataloader_status(test_dl, False)

        ds_name = test_dl.dataset.data_name
        ds_names.append(ds_name)
        start_time = time.time()
        # ds_count = len(test_dl.dataset)
        for batch_idx, (images, labels) in enumerate(test_dl):
            with torch.no_grad():
                images, labels = images.to(model.device, dtype=dtype), labels.to(model.device)
                # images = reverseDB2val(images)
                features = net.features(images)
                outputs = net.cls(features)
                _, max = torch.max(outputs, dim=-1)
                top1 += (labels == max).sum().item()
                total += labels.size(0)

                tmp_features.append(features)
                tmp_labels.append(labels)
        end_time = time.time()
        print(
            f'Reference Test throughput: {ds_names[j]} Time: {end_time - start_time:.3f}s Speed: {len(tmp_labels) / (end_time - start_time):.3f} item/s')

        tmp_features = torch.cat(tmp_features, dim=0)
        tmp_labels = torch.cat(tmp_labels, dim=0)
        indices = torch.randperm(tmp_features.size(0))[:show_count]
        tmp_features = tmp_features[indices]
        tmp_labels = tmp_labels[indices]
        if len(all_features) == 0:
            all_features = tmp_features[:show_count]
            all_labels = torch.tensor([j] * tmp_labels.size(0))[:show_count]
        else:
            all_features = torch.cat([all_features, tmp_features[:show_count]], dim=0)
            all_labels = torch.cat([all_labels, torch.tensor([j] * tmp_labels.size(0))[:show_count]], dim=0)

        test_dl = set_dataloader_status(test_dl, ds_status)
        top1acc = round(100 * top1 / total, 2)
        # if name in ['fl_digits','fl_officecaltech']:
        accs.append(top1acc)
    # elif name in ['fl_office31','fl_officehome']:
    #     accs.append(top5acc)
    if len(all_features) > 0:
        indices = torch.randperm(all_features.size(0))
        all_features = all_features[indices]
        all_labels = all_labels[indices]

    net.train(status)
    return accs, all_features, all_labels, ds_names


def train(method: BaseModel, private_dataset: BaseDataset,
          args: Namespace) -> None:
    method.N_CLASS = private_dataset.N_CLASS

    pri_train_loaders, test_loaders = private_dataset.get_data_loaders()

    method.trainloaders = pri_train_loaders
    if hasattr(method, 'ini'):
        method.ini()

    accs_dict = {}

    Epoch = args.eval_epoch
    for epoch_index in range(Epoch):
        method.epoch_index = epoch_index
        if hasattr(method, 'loc_update'):
            epoch_loc_loss_dict = method.loc_update(pri_train_loaders[0])

        # out domain
        accs, all_features, all_labels, ds_names = global_evaluate(method, test_loaders)
        out_domain_mean_acc = round(np.mean(accs, axis=0), 3)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        # print(accs)
        if len(all_features) > 0:
            indices = torch.randperm(all_features.size(0))
            all_features = all_features[indices]
            all_labels = all_labels[indices]

        upload_info = {}
        for j, dl in enumerate(test_loaders):
            upload_info.update({dl.dataset.data_name: accs[j]})
        upload_info.update({'Out domain avg': out_domain_mean_acc})

        # in domain
        accs, _, _, ds_names = global_evaluate(method, pri_train_loaders)
        in_domain_mean_acc = round(np.mean(accs, axis=0), 3)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        upload_info.update({'In domain avg': in_domain_mean_acc})
        print('The ' + str(epoch_index) + ' Evaluation Accuracy:', str(out_domain_mean_acc), 'Method:',
              method.args.method)

        print(upload_info)
        # upload_info.update({'outputs': all_outputs})
