import os
import sys
from typing import Tuple, List
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from utils import weights_folder


class Trainer:
    def __init__(self, model:torch.nn.Module, device:torch.device, experiment_name:str, class_weights:torch.Tensor = None):
        """
        :param model: Модель
        :param device: Устройство, на котором будет обучаться модель
        :param experiment_name: Название эксперимента. Нужно для загрузки весов и сохранения метрик.
        :param class_weights: Веса классов
        """
        self.device = device
        self.model = model.to(device)
        self.path_to_weights = os.path.join(weights_folder, f'{experiment_name}_model.pt')
        self.optimizer = Adam(list(self.model.parameters()), lr=1e-4)
        if any(class_weights):
            self.criterion = CrossEntropyLoss(weight=class_weights.float().to(self.device))
        else:
            self.criterion = CrossEntropyLoss()


    def train(self, train_loader:DataLoader, val_loader:DataLoader, num_epochs:int, save_model:bool=True) -> Tuple[List[float]]:
        """
        Обучение модели
        :param train_loader: Обучающие данные
        :param val_loader: Валидационные данные
        :param num_epochs: Количество эпох обучения
        :param save_model: Сохранять ли модель
        :return: Значения функции потерь, train accuracy и val accuracy за все эпохи обучения
        """
        loss_history = []
        train_history = []
        val_history = []
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}')
            self.model.train()  # Enter train mode
            loss_accum = 0
            correct = 0
            total = 0
            for batch_num, batch in enumerate(tqdm(train_loader)):
                prediction, y = self.get_prediction(batch)
                loss = self.compute_loss(prediction, y)
                loss_accum += loss
                accuracy, correct, total = self.compute_accuracy(prediction, y, correct, total)
                sys.stdout.write(f'\rLoss: {loss}. Train accuracy: {accuracy}')
                sys.stdout.flush()

            mean_loss = loss_accum / batch_num
            train_accuracy = float(correct) / total
            #val_accuracy = self.compute_val_accuracy(val_loader)

            loss_history.append(float(mean_loss))
            train_history.append(train_accuracy)
            #val_history.append(val_accuracy)

            #print(
               # "\nAverage loss: %f, Train accuracy: %f, Val accuracy: %f" % (mean_loss, train_accuracy, val_accuracy))
            if save_model:
                self.save_model()
        val_accuracy = self.compute_val_accuracy(val_loader)
        return loss_history, train_history, val_history

    def get_prediction(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получения предсказания модели
        :param batch: Мини-батч
        :return: Предсказания модели и истинные метки классов для всех семплов в мини-батче
        """
        ids = batch['input_ids'].to(self.device)
        mask = batch['attention_mask'].to(self.device)
        shop_ids = batch['shop_id'].to(torch.float32).to(self.device)
        images = batch['image'].to(self.device)
        y = batch['y'].to(self.device)
        prediction = softmax(self.model(desc=ids, attention_mask=mask, shop_id=shop_ids, img=images), dim=1)
        return prediction, y

    def compute_loss(self, prediction:torch.Tensor, y:torch.Tensor)-> float:
        """
        Вычисление функции потерь и оптимизация модели
        :param prediction: Предсказания модели
        :param y: Истинные метки классов
        :return: Значение функции потерь
        """
        loss = self.criterion(prediction, y)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_accuracy(self, prediction:torch.Tensor, y_true:torch.Tensor, correct:int, total:int) -> Tuple[float, int, int]:
        """
        Вычисление метрики accuracy на мини-батче
        :param prediction: Предсказания модели на данном мини-батче
        :param y_true: Истинные метки классов на данном мини-батче
        :param correct: Число правильных предсказаний за всю эпоху (без учета данного мини-батча)
        :param total: Общее число предсказаний за всю эпоху (без учета данного мини-батча)
        :return: Значение accuracy, число правильных предсказаний и общее число предсказаний за всю эпоху (с учетом данного мини-батча)
        """
        y_pred = torch.argmax(prediction, axis=1)
        correct += torch.sum(y_pred == y_true)
        total += y_true.shape[0]
        accuracy = correct / total
        return accuracy, correct, total

    def compute_val_accuracy(self, val_loader:DataLoader) -> float:
        """
        Вычисление метрики accuracy валидационной выборке
        :param val_loader: Валидационные данные
        :return: Значение метрики accuracy валидационной выборке
        """
        self.model.eval()
        correct = 0
        total = 0
        for batch_num, batch in enumerate(tqdm(val_loader)):
            prediction, y = self.get_prediction(batch)
            accuracy, correct, total = self.compute_accuracy(prediction, y, correct, total)
            sys.stdout.write(f'\rVal accuracy: {accuracy}')
            sys.stdout.flush()
        accuracy = float(correct) / total
        return accuracy

    def save_model(self):
        """
        Сохранение модели в файл
        """
        torch.save(self.model, self.path_to_weights)
        print(f'Model saved to {self.path_to_weights}')
