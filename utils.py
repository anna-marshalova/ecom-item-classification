import os
import re
import json
import torch
import pandas as pd
from typing import List
from importlib import import_module
from sklearn.metrics import classification_report

RANDOM_STATE = 42
#ROOT = '/content/drive/MyDrive/KazanExpress'
ROOT = '/content/drive/Marshalova_Anna'
metrics_folder = os.path.join(ROOT, 'metrics')
weights_folder = os.path.join(ROOT, 'weights')


def clear_html(text: str) -> str:
    """
    Очистка текста от HTML
    :param text: Описание товара
    :return: Описание товара, очищенное от HTML
    """
    html_tag = re.compile('<.*?/?>')
    return html_tag.sub(' ', text)


def get_description(text_fields: str, include_all_fields: bool = False) -> str:
    """
    Преобразование словаря в текст
    :param text_fields: Поле "text_fields" из датафрейма
    :param include_all_fields: Включать ли в текст описания дополнительные поля (фильтры, атрибуты и т.д.)
    :return: Текст, по которому будут классифицироваться товары
    """
    descr_dict = json.loads(text_fields)
    description = descr_dict['title']
    description += clear_html(descr_dict['description'])
    if include_all_fields:
        for key, value in descr_dict.items():
            if key not in ['title', 'description']:
                description += str(value)
    return description


def save_metrics(experiment_name: str, true: List[int], preds: List[int]):
    """
    Сохранение метрик в csv-файл
    :param experiment_name: Название эксперимента
    :param true: Истинные метки классов
    :param preds: Предсказанные метки классов
    """
    path_to_metrics = os.path.join(metrics_folder, f'{experiment_name}.csv')
    report = classification_report(true, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(path_to_metrics)
    print(f'Metrics saved to {path_to_metrics}')


def get_model(experiment_name: str, device: torch.device, num_labels: int, pretrained_model_name: str,
              load_weights: bool = True, **kwargs) -> torch.nn.Module:
    """
    Загрузка модели
    :param device: Устройство, на которое будет загружена модель
    :param experiment_name: Название эксперимента. Нужно для загрузки весов и архитектуры модели.
    :param num_labels: Количество классов
    :param pretrained_model_name: Название предобученной модели
    :param load_weights: Загрузить предобученные веса или обучать модель с нуля
    :return: Модель
    """""
    model_name = experiment_name+'_model'
    path_to_weights = os.path.join(weights_folder, f'{model_name}.pt')
    Model = getattr(import_module(model_name), 'Model')
    model = Model(pretrained_model_name=pretrained_model_name, num_labels=num_labels, **kwargs)
    if load_weights:
        print(f'Loading model from {path_to_weights}')
        model = torch.load(path_to_weights, map_location=device)
    return model
