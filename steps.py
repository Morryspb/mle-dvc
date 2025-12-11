# steps.py
#!/usr/bin/env python

import yaml
import pandas as pd
import joblib
import os
from scripts.data import get_data  # Импорт функции для загрузки данных
from scripts.fit_model import fit_model  # Импорт функции для обучения модели
from scripts.evaluate import evaluate_model  # Импорт функции для оценки модели


def load_params():
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
    return params

def main():
    params = load_params()

    # Шаг 1: загрузка данных
    print("Шаг 1: Загрузка данных...")
    data = get_data()  # Сохраняем результат в переменную data
    print("Данные загружены успешно.")
    print(type(data))  # проверить тип объекта
    #print(data.head())  # вывести первые строки DataFrame
    #return data

    # Шаг 2: обучение модели
    print("Шаг 2: Обучение модели...")
    fit_model(data)
    print("Модель обучена успешно.")
    

    # Шаг 3: оценка модели
    print("Шаг 3: Оценка модели...")
    evaluate_model()
    print("Модель оценена успешно.")

if __name__ == '__main__':
    main()
