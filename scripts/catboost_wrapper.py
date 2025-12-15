# scripts/catboost_wrapper.py

from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier
import numpy as np

class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    """
    Обёртка для CatBoostClassifier, полностью совместимая с scikit-learn API.
    Решает проблему с отсутствием метода __sklearn_tags__().
    """
    
    def __init__(self, **kwargs):
        """
        Инициализация обёртки.
        
        Parameters:
        -----------
        **kwargs : dict
            Параметры для CatBoostClassifier
        """
        self.kwargs = kwargs
        self.model = None
        self.classes_ = None
        self.n_classes_ = None
    
    def fit(self, X, y):
        """
        Обучение модели.
        
        Parameters:
        -----------
        X : array-like
            Признаки для обучения
        y : array-like
            Целевая переменная
        
        Returns:
        --------
        self : object
            Обученная модель
        """
        self.model = CatBoostClassifier(**self.kwargs)
        self.model.fit(X, y, verbose=False)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return self
    
    def predict(self, X):
        """
        Предсказание классов.
        
        Parameters:
        -----------
        X : array-like
            Признаки для предсказания
        
        Returns:
        --------
        predictions : array-like
            Предсказанные классы
        """
        if self.model is None:
            raise ValueError("Модель ещё не обучена. Вызовите fit() перед predict().")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей принадлежности к классам.
        
        Parameters:
        -----------
        X : array-like
            Признаки для предсказания
        
        Returns:
        --------
        probabilities : array-like
            Вероятности для каждого класса
        """
        if self.model is None:
            raise ValueError("Модель ещё не обучена. Вызовите fit() перед predict_proba().")
        return self.model.predict_proba(X)
    
    def get_params(self, deep=True):
        """
        Получение параметров модели.
        
        Parameters:
        -----------
        deep : bool
            Если True, возвращает параметры вложенных объектов
        
        Returns:
        --------
        params : dict
            Параметры модели
        """
        return self.kwargs
    
    def set_params(self, **params):
        """
        Установка параметров модели.
        
        Parameters:
        -----------
        **params : dict
            Параметры для установки
        
        Returns:
        --------
        self : object
            Модель с обновлёнными параметрами
        """
        self.kwargs.update(params)
        if self.model is not None:
            # Пересоздаём модель с новыми параметрами
            self.model = CatBoostClassifier(**self.kwargs)
        return self
