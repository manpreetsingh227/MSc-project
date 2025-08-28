from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from functools import partial

target_col = 'target'  
df = pd.read_parquet("/rds/general/user/ms2524/home/MSc-project/notebooks/oldmerged.parquet")
for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')
cat_cols = df.select_dtypes(include=['object', 'category']).columns
high_cardinality_cols = [col for col in cat_cols if df[col].nunique(dropna=False) > 30]
df = df.drop(columns=high_cardinality_cols)
X = df.drop(columns=[target_col])
y = df[target_col]
del df

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()


unique_weeks = sorted(X['WEEK_NUM'].unique())

n_weeks = len(unique_weeks)
test_weeks = unique_weeks[int(n_weeks * 0.9):]
valid_weeks = unique_weeks[int(n_weeks * 0.8):int(n_weeks * 0.9)]
train_weeks = unique_weeks[:int(n_weeks * 0.8)]


train_mask = X['WEEK_NUM'].isin(train_weeks)
valid_mask = X['WEEK_NUM'].isin(valid_weeks)
test_mask = X['WEEK_NUM'].isin(test_weeks)

# Split data
X_train, y_train = X[train_mask], y[train_mask]
X_valid, y_valid = X[valid_mask], y[valid_mask]
X_test, y_test = X[test_mask], y[test_mask]
del X
del y

cols_to_drop = ['WEEK_NUM', 'case_id']
X_train.drop(columns=cols_to_drop, inplace=True)
X_valid.drop(columns=cols_to_drop, inplace=True)
X_test.drop(columns=cols_to_drop, inplace=True)



print(f"Train: {X_train.shape[0]}, Validation: {X_valid.shape[0]}, Test: {X_test.shape[0]}")

model = CatBoostClassifier(
    task_type='GPU',
    devices='0',
    iterations=1000,
    auto_class_weights='Balanced', 
    eval_metric='PRAUC',           
    random_seed=42,
    early_stopping_rounds=50,
    verbose=200
)

# === 4. Train Model ===
model.fit(
    X_train, y_train,
    cat_features=cat_cols,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

best_iteration = model.get_best_iteration()
test_probs = model.predict_proba(X_test)[:, 1]


feature_importances = model.get_feature_importance(Pool(X_train, label=y_train, cat_features=cat_cols))
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

importance_df.to_csv('/rds/general/user/ms2524/home/MSc-project/notebooks/importance.csv', index=False)
