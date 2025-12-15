import pandas as pd
import sklearn
import category_encoders
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import catboost
from catboost import CatBoostClassifier
import joblib

#проверка версий
print(sklearn.__version__)
print(category_encoders.__version__)

data = pd.read_csv('data/initial_data.csv')
print(data.head())
def fit_model(data):
    # обучение модели
    cat_features = data.select_dtypes(include='object')
    print(cat_features)
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = data.select_dtypes(['float'])

    preprocessor = ColumnTransformer(
        [
            ('binary', OneHotEncoder(drop='if_binary'), binary_cat_features.columns.tolist()),
            ('cat', CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    model = CatBoostClassifier(auto_class_weights='Balanced')

    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    
    X = data.drop('target', axis=1)  # предполагая, что 'target' это целевая переменная
    y = data['target']
    
    pipeline.fit(X, y)
    print('Модель обучена')
    
    # Сохранение модели
    joblib.dump(pipeline, 'models/model.pkl')
    
    return pipeline

if __name__ == "__main__":
    fit_model(data)
