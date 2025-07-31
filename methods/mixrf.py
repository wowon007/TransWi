import math
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np

from methods.utils.basemodel import BaseModel


def get_param_groups(model, weight_decay=0.01):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]


class MixRf(BaseModel):
    NAME = 'Mixrf'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, net, args, transform):
        super(MixRf, self).__init__(net, args, transform)
        self.alpha = getattr(args, 'alpha', 1)
        self.prob_mix_weight = getattr(args, 'prob_mix_weight', 0.5)
        self.theta = getattr(args, 'theta', 0.02)
        self.mix_loss_weight = getattr(args, 'mix_loss_weight', 0.5)
        self.opt_name = getattr(args, 'opt_name', 'adamw')

    def ini(self):
        self.criterion = nn.CrossEntropyLoss()
        self.pro_calc_method = nn.Softmax(dim=-1)
        # self.pro_calc_method = Sparsemax(dim=-1)
        self.param_scale = 1.4
        self.init_centers(net=self.net)
        if self.opt_name == 'adamw':
            self.optimizer = optim.AdamW(get_param_groups(model=self.net, weight_decay=1e-5), lr=self.lr,
                                         weight_decay=1e-5, amsgrad=True)
        elif self.opt_name == 'adam':
            self.optimizer = optim.Adam(get_param_groups(model=self.net, weight_decay=1e-5), lr=self.lr,
                                        weight_decay=1e-5)
        elif self.opt_name == 'sgd':
            self.optimizer = optim.SGD(get_param_groups(model=self.net, weight_decay=1e-5), lr=self.lr, momentum=0.8,
                                       weight_decay=1e-5)
        else:
            raise ValueError(f"Unsupported optimizer: {self.opt_name}")
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                       T_max=int(self.local_epoch * self.eval_epoch),
                                                                       eta_min=1e-7)

        self.iter = 0

    def init_centers(self, net):
        self.al_layers = ['feature']
        num_classes = self.N_CLASS
        net.al_layers = self.al_layers

        net.al_centers = {}
        layer_name = net.al_layers[-1]
        shape = (net.cls.in_features,)
        net.al_centers[layer_name] = {}
        param_name = f'{layer_name}_centers'
        net.al_centers[layer_name]['centers'] = nn.Parameter(
            self.param_scale * torch.rand((num_classes,) + shape).to(self.device, dtype=self.dtype),
            requires_grad=True)
        # net.al_centers[layer_name]['centers'].data = net.cls.weight.data

        net.al_centers[layer_name]['labels'] = torch.tensor([[i] for i in range(num_classes)]).view(-1).to(
            self.device)
        net.register_parameter(param_name, net.al_centers[layer_name]['centers'])


    def loc_update(self, train_loader):
        self._train_net(train_loader)
        return None

    def _train_net(self, train_loader):
        net = self.net.to(self.device)
        net.train()

        train_loader_len = len(train_loader)
        tmp_loss_dict = {}

        with tqdm(total=self.local_epoch, ncols=200) as _tqdm:  
            for epoch in range(self.local_epoch):
                _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, self.local_epoch)) 
                loss_dict = {'loss': 0, 'ac_loss': 0, 'bn_mean': 0, 'bn_var': 0}
                for batch_idx, (images, labels) in enumerate(train_loader):
                    ac_loss = 0
                    loss_mix = 0
                    uni_loss = 0
                    uni_loss_1 = 0
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    loss_mix, tmp_loss_dict, mixed_features, output_features, shuffled_label, weight = mixed_loss_cls_based(
                        net=net,
                        data=images,
                        label=labels,
                        alpha=self.alpha,
                        num_classes=self.N_CLASS,
                        prob_mix_weight=self.prob_mix_weight)
                    output_features = net.features(images)
                    outputs = net.cls(output_features)
                    ac_loss = self.criterion(outputs, labels)
                    centers = net.al_centers[net.al_layers[-1]]['centers'][labels]

                    output_feature_with_mixed = output_features
                    uni_loss, mm_loss, kl_loss = calc_euc_uniform_dis_loss(output_feature_with_mixed.squeeze(),
                                                                            centers.squeeze(),
                                                                            std=self.theta,
                                                                            mmd_weight=1,
                                                                            dim=0)

                    loss = ac_loss + self.mix_loss_weight * loss_mix + (1 - self.mix_loss_weight) * (
                            uni_loss_1 + uni_loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_dict['loss'] += loss.item() / train_loader_len
                    loss_dict['ac_loss'] += ac_loss.item() / train_loader_len

                    for key in tmp_loss_dict:
                        if key in loss_dict.keys():
                            loss_dict[key] += tmp_loss_dict[key] / train_loader_len
                        else:
                            loss_dict[key] = tmp_loss_dict[key] / train_loader_len

                desc_str = ['{} = {:.3f}'.format(key, loss_dict[key]) for key in
                            loss_dict.keys()]
                _tqdm.postfix = f"{', '.join(desc_str)}"
                _tqdm.update(1)
                self.iter += 1
                self.lr_scheduler.step()




import random


def frequency_shift(spectrogram, shift):
    return torch.roll(spectrogram, shifts=shift, dims=1)


def patch_cutmix_freq_aug(batch, labels, alpha=1.0, num_patches=3, shift_prob=0.5):
    B, C, T, F = batch.size()
    indices = torch.randperm(B)
    shuffled_batch = batch[indices]
    shuffled_labels = labels[indices]
    mixed_labels = torch.zeros_like(labels, device=labels.device, dtype=torch.float32)

    mixed_batch = batch.clone()
    weight = torch.zeros((B,))

    for i in range(B):

        shift_val = random.randint(-F // 8, F // 8)
        mixed_batch[i] = frequency_shift(mixed_batch[i], shift_val)
        shuffled_batch[i] = frequency_shift(shuffled_batch[i], -shift_val)
        mask = torch.zeros((T, F), dtype=torch.bool)
        lam_patch = np.random.beta(alpha, alpha)
        areas = np.random.dirichlet(np.ones(num_patches)) * lam_patch

        for j in range(num_patches):
            patch_frac = np.sqrt(areas[j])  
            freq_cut = max(1, int(F * patch_frac))
            time_cut = max(1, int(T * patch_frac))

            freq_start = np.random.randint(0, F - freq_cut + 1)
            time_start = np.random.randint(0, T - time_cut + 1)

            mixed_batch[i, :, time_start:time_start + time_cut, freq_start:freq_start + freq_cut] = \
                shuffled_batch[i, :, time_start:time_start + time_cut, freq_start:freq_start + freq_cut]

            mask[time_start:time_start + time_cut, freq_start:freq_start + freq_cut] = True

        mixed_area = mask.sum().item()
        lam = 1.0 - mixed_area / (F * T)
        weight[i] = lam


        mixed_labels[i] = lam * labels[i] + (1 - lam) * shuffled_labels[i]

    return mixed_batch, mixed_labels, weight, indices



def mixed_loss_cls_based(net, data, label, alpha, num_classes, prob_mix_weight=0.45):
    # generate label
    oh_label = torch.nn.functional.one_hot(label, num_classes=num_classes)
    with torch.no_grad():
        mixed_data, oh_mixed_label, weight, indices = patch_cutmix_freq_aug(data, oh_label, alpha=alpha, shift_prob=0.5)
    output_features = None
    mixed_outputs = net(mixed_data)
    prob_mix_weight = prob_mix_weight
    mixed_outputs_prob = torch.softmax(mixed_outputs, dim=-1)
    with torch.no_grad():
        tmp_mixed_outputs = mixed_outputs_prob * oh_label + mixed_outputs_prob * oh_label[indices]
        tmp_oh_mixed_label = (1 - prob_mix_weight) * tmp_mixed_outputs + prob_mix_weight * oh_mixed_label
    loss_mixed = torch.sum(- tmp_oh_mixed_label * torch.log(mixed_outputs_prob), dim=-1).mean()
    loss_mixed_total = loss_mixed

    loss_dic = {
        'loss_mixed_total': loss_mixed_total,
        'loss_mixed': loss_mixed,
    }
    return loss_mixed_total, loss_dic, None, output_features, None, None



def fit_normal_distribution(distances, dim=0):
    mean = torch.mean(distances, dim=dim, keepdim=True)
    std = torch.std(distances, dim=dim, keepdim=True)
    return mean, std


def calc_euc_uniform_dis_loss(features, prompt_pool, labels=None, std=0.01, sigma=0.5, mmd_weight=1.0, dim=0, eps=1e-5):
    if labels is None:
        prompts = prompt_pool
    else:
        prompts = prompt_pool[labels]

    diff = (features - prompts) / (torch.mean(torch.abs(prompts)))


    target_std = std

    mean, std = fit_normal_distribution(diff, dim=dim)

    target_std = target_std

    kl_diss = torch.mean(
        calc_js_divergence_normal(mean=mean, std=std, target_mean=0.0, target_std=target_std, dim=dim, eps=eps))
    mmd_loss_value = 0

    tmp_diff = diff
    if dim == 0:
        tmp_diff = diff.T
    n_rows, n_features = tmp_diff.shape
    target_samples = torch.normal(0, target_std, size=(n_rows, n_features)).to(tmp_diff.device,
                                                                               dtype=tmp_diff.dtype)
    mmd_loss_value = mk_mmd_loss(tmp_diff, target_samples)
    return mmd_weight * mmd_loss_value + 1 * kl_diss, mmd_loss_value, kl_diss



def calc_js_divergence_normal(mean, std, target_mean=0.0, target_std=1.0, dim=0, eps=1e-5):
    target_std = torch.ones_like(std) * target_std

    def kl_divergence(mean_p, std_p, mean_q, std_q):
        return torch.log(std_q / std_p + eps) + (std_p ** 2 + (mean_p - mean_q) ** 2) / (2 * std_q ** 2) - 0.5

    mean_m = (mean + target_mean) / 2
    std_m = (std + target_std) / 2

    kl_p_m = kl_divergence(mean, std, mean_m, std_m)
    kl_q_m = kl_divergence(target_mean, target_std, mean_m, std_m)

    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div


def mmd_loss(diff, target_mean=0.0, target_std=1.0, kernel='rbf', sigma: float | None = None, dim=0):
    tmp_diff = diff
    if dim == 0:
        tmp_diff = diff.T

    n_rows, n_features = tmp_diff.shape

    target_samples = torch.normal(target_mean, target_std, size=(n_rows, n_features)).to(tmp_diff.device,
                                                                                         dtype=tmp_diff.dtype)

    if kernel == 'rbf':
        def rbf_kernel(x, y, sigma=1.0):
            x_dtype = x.dtype
            y_dtype = y.dtype
            if x_dtype != torch.float32 or y_dtype != torch.float32:
                x = x.to(torch.float32)
                y = y.to(torch.float32)
            dist = torch.cdist(x, y, p=2)
            dist = dist.to(dtype=x_dtype)
            return torch.exp(-dist ** 2 / (2 * sigma ** 2))

        kernel_fn = rbf_kernel
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    if sigma is None:
        dist = torch.cdist(diff, diff, p=2)
        sigma = torch.median(dist) / torch.sqrt(torch.tensor(2.0))

    xx = kernel_fn(tmp_diff, tmp_diff, sigma)  # (n_rows, n_rows)
    yy = kernel_fn(target_samples, target_samples, sigma)  # (n_rows, n_rows)
    xy = kernel_fn(tmp_diff, target_samples, sigma)  # (n_rows, n_rows)

    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd


def compute_kernel(x, y, kernel_type='gaussian', kernel_param=1.0):
    if kernel_type == 'gaussian':
        x_norm = (x ** 2).sum(dim=-1, keepdim=True) 
        y_norm = (y ** 2).sum(dim=-1, keepdim=True) 
        xy = torch.matmul(x, y.t())
        pairwise_distance = x_norm + y_norm.t() - 2 * xy 
        kernel_matrix = torch.exp(-pairwise_distance / (2 * kernel_param ** 2))  
    elif kernel_type == 'linear':
        kernel_matrix = torch.matmul(x, y.t())
    else:
        raise ValueError("Unsupported kernel type.")

    return kernel_matrix


def mk_mmd_loss(x, y, kernel_types=('gaussian', 'gaussian', 'gaussian', 'gaussian', 'gaussian'),
                kernel_params=(0.1, 0.5, 1.0, 2.0, 5.0)):
    batch_size = x.size(0)
    n_kernels = len(kernel_types)

    xx_kernels = [compute_kernel(x, x, kernel_type, kernel_param) for kernel_type, kernel_param in
                  zip(kernel_types, kernel_params)]
    yy_kernels = [compute_kernel(y, y, kernel_type, kernel_param) for kernel_type, kernel_param in
                  zip(kernel_types, kernel_params)]
    xy_kernels = [compute_kernel(x, y, kernel_type, kernel_param) for kernel_type, kernel_param in
                  zip(kernel_types, kernel_params)]

    mk_mmd = 0.0
    for i in range(n_kernels):
        xx = xx_kernels[i]
        yy = yy_kernels[i]
        xy = xy_kernels[i]
        mmd = torch.mean(xx) - 2 * torch.mean(xy) + torch.mean(yy)
        mk_mmd += torch.sqrt(torch.max(torch.tensor(0.0), mmd))

    mk_mmd /= n_kernels

    return mk_mmd

