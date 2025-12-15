# scripts/evaluate.py

# 1 — Импорты
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import json
import os
import yaml

def evaluate_model():
    """
    Функция оценки качества обученной модели.
    
    Выполняет следующие шаги:
    1. Загружает данные и параметры
    2. Разделяет данные на обучающую и тестовую выборки
    3. Загружает обученную модель
    4. Делает предсказания на тестовой выборке
    5. Вычисляет метрики качества
    6. Сохраняет результаты в JSON-файл
    """
    
    # 3.1 — Загрузка гиперпараметров из params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
    print("Параметры загружены из params.yaml")
    
    # 3.2 — Загрузка данных
    data = pd.read_csv('data/initial_data.csv')
    print("Данные успешно загружены")
    print(f"Форма датасета: {data.shape}")
    
    # 3.3 — Разделение данных на признаки и целевую переменную
    target_col = params.get('target_col', 'target')
    X = data.drop(columns=target_col)
    y = data[target_col]
    
    # ВАЖНО: Используем те же параметры разделения, что и при обучении
    # Если вы хотите оценить модель на отложенной выборке, убедитесь,
    # что используете тот же random_state, что и при обучении
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Сохраняем пропорции классов в обеих выборках
    )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Распределение классов в тесте:\n{y_test.value_counts()}")
    
    # 3.4 — Загрузка обученной модели
    model_path = 'models/fitted_model.pkl'
    try:
        pipeline = joblib.load(model_path)
        print(f"Модель успешно загружена из {model_path}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        raise
    
    # 3.5 — Предсказания на тестовой выборке
    print("Выполнение предсказаний...")
    try:
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        print("Предсказания выполнены успешно")
    except Exception as e:
        print(f"Ошибка при выполнении предсказаний: {e}")
        raise
    
    # 3.6 — Вычисление метрик качества
    print("Вычисление метрик...")
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # 3.7 — Сохранение результатов
    os.makedirs('cv_results', exist_ok=True)
    results_path = 'cv_results/cv_res.json'
    
    results = {
        'accuracy': float(accuracy),  # Преобразуем в float для корректной сериализации
        'roc_auc': float(roc_auc),
        'classification_report': report,
        'test_size': len(y_test),
        'train_size': len(y_train)
    }
    
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Результаты успешно сохранены в {results_path}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")
        raise
    
    return results

# 4 — Защищённый вызов главной функции
if __name__ == '__main__':
    evaluate_model()

