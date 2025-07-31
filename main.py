import os
import sys
import socket
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import warnings

warnings.filterwarnings("ignore")

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')
from datasets import DS_NAMES
from methods import get_all_methods, get_method
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_dataset
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.utils import save_files_to_date_time_folder
import setproctitle

import torch
import uuid
import datetime
import inspect


def parse_args():
    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=None, help='The Device Id for Experiment')
    parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch Count Before Evaluation')
    parser.add_argument('--eval_epoch', type=int, default=20, help='The Evaluation Epoch Count')

    parser.add_argument('--seed', type=int, default=0, help='The random seed.')

    parser.add_argument('--method', type=str, default='cross_entropy',
                        help='Method name.', choices=get_all_methods())  ## cross_entropy  mixrf  wiflexformer
    parser.add_argument('--structure', type=str, default='homogeneity')
    parser.add_argument('--dataset', type=str, default='isac',  # dataset name
                        choices=DS_NAMES, help='Which scenario to perform experiments on.')

    parser.add_argument('--learning_decay', type=bool, default=False, help='The Option for Learning Rate Decay')
    parser.add_argument('--beta', type=float, default=0.4, help='momentum update beta')

    torch.set_num_threads(8)
    add_management_args(parser)
    args = parser.parse_args()

    return args


def main(args=None, param_dict=None):
    if args is None:
        args = parse_args()

    best = best_args[param_dict.get('dataset', args.dataset)][param_dict.get('method', args.method)]

    for key, value in best.items():
        setattr(args, key, value)

    if param_dict is not None:
        for key, value in param_dict.items():
            setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    priv_dataset = get_dataset(args)

    net_backbone = vars(args).get('backbone', None)
    if net_backbone != None:
        net_backbone = net_backbone
    else:
        net_backbone = 'resnet18'

    backbones_list = priv_dataset.get_backbone(net_backbone, args=args)

    method = get_method(backbones_list, args, priv_dataset.get_transform())
    args.arch = method.net.name

    print('{}_{}_{}_{}'.format(args.method, net_backbone, args.dataset, args.local_epoch))
    setproctitle.setproctitle(
        '{}_{}_{}_{}'.format(args.method, net_backbone, args.dataset, args.local_epoch))


    base_path = os.getcwd()
    src_paths = [inspect.getfile(backbones_list.__class__), inspect.getfile(method.__class__), __file__.split("/")[-1]
                 ]
    src_paths = [os.path.relpath(src_path, base_path) for src_path in src_paths]
    save_files_to_date_time_folder(base_path='file_snapshot', src_paths=src_paths)

    train(method, priv_dataset, args)



if __name__ == '__main__':


    ds = ['widar']
    for d in ds:
        for i in range(2):
            param_dict = {
                'offline': False,
                'backbone': 'transwi',
                'dataset': d,  # 'gecro', 'widar', 'isac'
                'method': 'mixrf', 
                'alpha': 1,
                'lr': 1e-3,
                'prob_mix_weight': 0.5,
                'mix_loss_weight': 0.5,
                'seed': i,
            }
            main(param_dict=param_dict)
