import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
from typing import Tuple, Dict, Any
from sklearn.preprocessing import OneHotEncoder

import torch
from torchvision import transforms
from transformers import AutoTokenizer

from utils import get_description

torch.set_default_dtype(torch.float32)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename: str, pretrained_model_name: str, mode: str = 'train'):
        """
        :param filename: Название файла с датасетом
        :param pretrained_model_name: Название предобученной языковой модели
        :param mode: Режим, для которого создается датасет. По умолчанию train (датасет для обучения)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.one_hot_enc = OneHotEncoder()
        self.max_length = 500
        self.mode = mode
        self.path_to_images = os.path.join('images', self.mode)
        self.prepare_dataset(filename)
        self.num_labels = len(self.category2class)

    def prepare_dataset(self, filename: str):
        """
        Подготовка данных
        :param filename: Название файла с датасетом
        """
        print('Praparing dataset...')
        self.df = pd.read_parquet(filename).reset_index()
        self.df['description'] = pd.Series(map(get_description, tqdm(self.df.text_fields))).fillna('')
        if self.mode == 'test':
                self.df['category_id'] = pd.Series(np.zeros(len(self.df)))
        self.category2class = self.numerize_categorical(self.df.category_id)
        self.shop_id2categorical = self.numerize_categorical(self.df.shop_id)
        self.one_hot_enc.fit(np.asarray(list(self.shop_id2categorical.values())).reshape(-1, 1))

    def numerize_categorical(self, column: pd.Series) -> Dict[Any, int]:
        """
        Нумерация категориальных признаков последовательностью от 0 до N
        :param column: Столбец датафрейма
        :return: Словарь вида 'признак:номер признака'
        """
        unique = sorted(column.unique())
        return dict(zip(unique, range(len(unique))))

    def vectorize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Претокенизация текста
        :param text: Текст с описанием товара
        :return: Инпут для языковой модели и маска
        """
        inputs = self.tokenizer(
            text,
            return_attention_mask=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

    def get_image(self, product_id: int) -> torch.Tensor:
        """
        Векторизация изображения
        :product_id: Идентификатор товара
        :return: Инпут для модели компьютерного зрения
        """
        path = os.path.join(self.path_to_images, f'{product_id}.jpg')
        img = np.array(Image.open(path))
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        img = transforms.Resize((224, 224))(img)
        return img

    def get_shop_id(self, shop_id: int) -> torch.Tensor:
        """
          One-hot векторизация идентификатора магазина
          :shop_id: Идентификатор магазина
          :return: Инпут для модели, работающей с категориальными признаками
          """
        shop_id_num = self.shop_id2categorical[shop_id]
        one_hot_shop_id = self.one_hot_enc.transform(np.asarray(shop_id_num).reshape(-1, 1))
        return torch.Tensor(one_hot_shop_id.todense()).reshape(-1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if isinstance(idx, int):
            row = self.df.loc[idx]
            ids, mask = self.vectorize(row.description)
            y = self.category2class[row.category_id]
            shop_id = self.get_shop_id(row.shop_id)
            image = self.get_image(row.product_id)
            return {'input_ids': ids, 'attention_mask': mask, 'shop_id': shop_id, 'image': image, 'y': y, 'product_id':row.product_id}