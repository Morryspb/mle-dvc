# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import json
import os

def evaluate_model():
    # Загрузка данных
    data = pd.read_csv('data/initial_data.csv')

    # Разделение данных
    X = data.drop(columns='target')
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Загрузка модели
    pipeline = joblib.load('models/fitted_model.pkl')

    # Предсказания
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Вычисление метрик
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Сохранение результатов
    os.makedirs('cv_results', exist_ok=True)
    with open('cv_results/cv_res.json', 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': report
        }, f, indent=4)

if __name__ == '__main__':
    evaluate_model()
