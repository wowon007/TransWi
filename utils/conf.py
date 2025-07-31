import random
import torch
import numpy as np


def get_device(device_id=None) -> torch.device:
    if device_id is not None:
        return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    import subprocess
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_used = [int(x) for x in result.strip().split('\n')]
        min_memory_gpu = memory_used.index(min(memory_used))
        return torch.device(f"cuda:{min_memory_gpu}")
    except Exception as e:
        print(f"Warning: Could not determine GPU memory usage, falling back to cuda:0. Error: {e}")
        return torch.device("cuda:0")


def data_path_widar() -> str:
    return '../datas/structured-unintegrated-processed/Widar3'

def data_path_gecro() -> str:
    return '../datas/structured-unintegrated-processed/GeCro'

def data_path_isac() -> str:
    return '../datas/structured-unintegrated-processed/ISAC'


def base_path() -> str:
    return './data/'


def checkpoint_path() -> str:
    return './checkpoint/'


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
