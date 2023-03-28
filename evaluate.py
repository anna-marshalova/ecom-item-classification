import sys
import pandas as pd
from tqdm.notebook import tqdm
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.metrics import classification_report, f1_score


def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[List[int]]:
    """
    Формирование classification report для заданной модели на заданных данных
    :param model: Модель
    :param data_loader: Данные для оценки
    :param device: Устройство, на котором будет происходить инференс модели
    :return: Истинные и предсказанные метки классов
    """
    model.to(device)
    true = []
    preds = []
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(data_loader)):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            shop_ids = batch['shop_id'].unsqueeze(1).to(torch.float32).to(device)
            images = batch['image'].to(device)
            y = batch['y'].to(device)
            prediction = softmax(model(desc=ids, attention_mask=mask, shop_id=shop_ids, img=images), dim=1)
            y_pred = torch.argmax(prediction, axis=1)
            preds.extend(list(y_pred.cpu()))
            true.extend(list(y.cpu().numpy()))
            sys.stdout.write(f'\rF1 for batch {batch_num}: {f1_score(y.cpu(), y_pred.cpu(), average="weighted")}')
            sys.stdout.flush()
    print(f'Total F1 score: {f1_score(true, preds, average="weighted")}')
    return true, preds
