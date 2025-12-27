# scripts/fit.py

# 1 — Импорты
import pandas as pd
import sklearn
import category_encoders
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import os
import yaml

# 2 — Проверка версий библиотек (для отладки)
print(f"Версия sklearn: {sklearn.__version__}")
print(f"Версия category_encoders: {category_encoders.__version__}")


def fit_model():
    """
    Функция обучения модели машинного обучения.
    """
    # 3.1 — Загрузка гиперпараметров из params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
    print("Параметры загружены из params.yaml")


    # 3.2 — Загрузка данных
    data = pd.read_csv('data/initial_data.csv')
    print("Данные успешно загрузились:")
    print(data.head())

    # 3.7 — Разделение данных (переставлено с позиции после 3.3)
    # Здесь: сразу после загрузки данных, до подготовки признаков
    target_col = params.get('target_col', 'target')
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    print(f"Форма обучающей выборки: X={X.shape}, y={y.shape}")


    # 3.3 — Подготовка признаков (теперь работает с X, где target уже удалена)
    cat_features = X.select_dtypes(include='object')
    print(f"Категориальные признаки:\n{cat_features.columns.tolist()}")


    cat_features_list = cat_features.columns.tolist()
    num_features = X.select_dtypes(['float', 'int'])  # берём из X, а не из data!
    num_features_list = num_features.columns.tolist()


    print(f"Категориальные признаки для OneHotEncoder:\n{cat_features_list}")
    print(f"Числовые признаки:\n{num_features_list}")


    # 3.5 — Создание модели с использованием параметров из params.yaml
    model_params = params['model']
    estimator_name = model_params['estimator']

    if estimator_name == 'logistic_regression':
        model = LogisticRegression(
            C=model_params['C'],
            penalty=model_params['penalty'],
            solver=model_params['solver'],
            random_state=model_params.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unsupported model estimator: {estimator_name}")

    print(f"Модель инициализирована: {estimator_name} с параметрами C={model_params['C']}, penalty={model_params['penalty']}")


    # 3.6 — Создание пайплайна
    preprocessor = ColumnTransformer(
        [
            ('cat', OneHotEncoder(drop='if_binary', sparse_output=False), cat_features_list),
            ('num', StandardScaler(), num_features_list)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 3.8 — Обучение модели (осталось на своём месте)
    pipeline.fit(X, y)
    print("Модель успешно обучена")


    # 3.9 — Сохранение модели
    os.makedirs('models', exist_ok=True)
    model_path = 'models/fitted_model.pkl'
    joblib.dump(pipeline, model_path)
    print(f"Модель успешно сохранена в {model_path}")


    return pipeline


if __name__ == "__main__":
    fit_model()
#target дропаем из данных ДО выделения признаков, потому что целевая колонка может попасть в эти признаки (бинарные) и затем просто выпилиться из данных, и тогда код обучения ее тупо не найдет