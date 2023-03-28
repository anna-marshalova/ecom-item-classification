import pandas as pd
from typing import Dict
from tqdm.notebook import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax


def predict(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, category2class:Dict[str,int]) -> pd.DataFrame:
    """
    Формирование датасета с предсказаниями для тестовых данных
    :param model: Модель
    :param data_loader: Тестовые данные
    :param device: Устройство, на котором будет происходить инференс модели
    :param category2class: Словарь, преобразующий номера категорий в номера классов для обучения
    :return: DataFrame с предсказаниями
    """
    model.to(device)
    product_ids = []
    preds = []
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(data_loader)):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            shop_ids = batch['shop_id'].unsqueeze(1).to(torch.float32).to(device)
            images = batch['image'].to(device)
            p_ids = batch['product_id']
            prediction = softmax(model(desc=ids, attention_mask=mask, shop_id=shop_ids, img=images), dim=1)
            y_pred = torch.argmax(prediction, axis=1)
            preds.extend(list(y_pred.cpu().numpy()))
            product_ids.extend(list(p_ids.numpy()))
    class2category = dict(zip(category2class.values(), category2class.keys()))
    category_ids = map(class2category.get, preds)
    result_df = pd.DataFrame({'product_id':product_ids,'predicted_category_id':category_ids})
    return result_df

