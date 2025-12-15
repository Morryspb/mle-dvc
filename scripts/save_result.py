import pandas as pd
import joblib

# например
os.makedirs('data-directory', exist_ok=True) # создание директории, если её ещё нет
data.to_csv('path/to/data/file')

# или
os.makedirs('model-directory', exist_ok=True) # создание директории, если её ещё нет
with open('path/to/model/file', 'wb') as fd:
    joblib.dump(model, fd)