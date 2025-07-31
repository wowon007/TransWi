import os
import time
import torch
import numpy as np
import shutil
from datetime import datetime


def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def check_date_dir(target_path):
    check_dir_exist(target_path)
    root_path = "{}".format(target_path)
    local_time = time.localtime()
    tmp_path = time.strftime("%Y%m%d", local_time)
    root_path = root_path + "/{}/".format(tmp_path)
    check_dir_exist(root_path)
    return root_path, local_time


def check_dir_exist(path, mkdir=True):
    if '.' == path[0]:
        path = path[1:]
        tmp = '.'
    else:
        tmp = ''
    pathList = path.split('/')[1:]
    for p in pathList:
        tmp += '/' + p
        tmp = tmp.strip()
        if not os.path.exists(tmp) and '.' not in p:
            if mkdir:
                os.mkdir(tmp)


def translate_to_numpy(data, dtype=None):
    if type(data) == list:
        data = np.array(data)

    if type(data) != np.ndarray:
        try:
            data = data.numpy()
        except Exception:
            if data.dtype == torch.bfloat16:
                data = data.to(torch.float32)
            data = data.cpu().numpy()

    if dtype is not None:
        data = data.astype(dtype)
    return data


def find_case_insensitive_file(path):
    if os.path.isfile(path):
        return path

    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path).lower()

    if not os.path.isdir(dir_name):
        return None

    for fname in os.listdir(dir_name):
        if fname.lower() == base_name:
            return os.path.join(dir_name, fname)
    return None


def save_files_to_date_time_folder(base_path, src_paths):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    base_output_dir = os.path.join(os.getcwd(), base_path, date_str, time_str)

    for src_path in src_paths:
        actual_path = find_case_insensitive_file(src_path)

        if not actual_path:
            continue

        rel_path = os.path.splitdrive(src_path)[1].lstrip(os.sep)
        dest_path = os.path.join(base_output_dir, rel_path)

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(actual_path, dest_path)

