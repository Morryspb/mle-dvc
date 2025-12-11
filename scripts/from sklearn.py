from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

# Предсказание на тестовой выборке
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Вероятности для ROC-AUC

# Оценка модели по заданным метрикам
metrics = params['metrics']
results = {}

if 'f1' in metrics:
    f1 = f1_score(y_test, y_pred)
    results['f1'] = f1
    print(f'F1-score: {f1:.4f}')

if 'roc_auc' in metrics:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    results['roc_auc'] = roc_auc
    print(f'ROC AUC: {roc_auc:.4f}')

print('Оценка модели завершена. Результаты:', results)
