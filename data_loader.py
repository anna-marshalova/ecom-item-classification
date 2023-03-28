import numpy as np
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from utils import RANDOM_STATE


def get_train_loaders(dataset: Dataset, batch_size: int = 64, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Разделение данных на train и val, разбиение на батчи
    :param dataset: torch.utils.data.Dataset Обучающий датасет
    :param batch_size: Размер мини-батча
    :param num_workers: Количество потоков параллельной загрузки данных в модель
    :return: Загрузчики обучающих и валидационных данных
    """
    val_split = int(np.floor(0.2 * len(dataset)))
    indices = list(range(len(dataset)))
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[val_split:])
    val_sampler = SubsetRandomSampler(indices[:val_split])
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
    return train_loader, val_loader

