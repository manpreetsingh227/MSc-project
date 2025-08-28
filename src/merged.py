import pandas as pd
import numpy as np
import gc
from catboost import CatBoostClassifier

base_path = '/rds/general/user/ms2524/home/amexproject/parquet_files/train/train_'
base_cba = pd.read_parquet(base_path + "base.parquet", columns=['case_id', 'WEEK_NUM', 'target', 'date_decision'])
base_cba['date_decision'] = pd.to_datetime(base_cba['date_decision'], errors='coerce')

def downcast_numeric(df):
    for col in df.select_dtypes(include=["int", "float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="unsigned" if df[col].min() >= 0 else "integer")
    return df

def convert_columns(df):
    skip_cols = {'case_id', 'WEEK_NUM', 'target', 'date_decision'}
    for col in df.columns:
        if col in skip_cols:
            continue
        col_upper = col.upper()
        if col_upper.endswith('P'):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif col_upper.endswith('A'):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        elif col_upper.endswith('D'):
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif df[col].dtype == 'object' or col_upper.endswith('M'):
            df[col] = df[col].fillna('Unknown').astype('category')
    return df

def convert_dates_to_days_from_decision(df, decision_col='date_decision', drop_original=True):
    if not pd.api.types.is_datetime64_any_dtype(df[decision_col]):
        df[decision_col] = pd.to_datetime(df[decision_col], errors='coerce')
    date_cols = [col for col in df.columns if col.endswith('D') and col != decision_col]
    for col in date_cols:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f"{col}_days_from_decision"] = (df[decision_col] - df[col]).dt.days
        if drop_original:
            df.drop(columns=col, inplace=True)
    return df

def load_and_merge(name, condition=None, drop_cols=None):
    df = pd.read_parquet(base_path + f"{name}.parquet")
    if condition:
        df = df.query(condition)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = pd.merge(base_cba, df, how="left", on="case_id")
    return df

# Transaction data
deposit_1    = load_and_merge("deposit_1",    "num_group1 == 0", ["num_group1"])
debitcard_1  = load_and_merge("debitcard_1",  "num_group1 == 0", ["num_group1"])
other_1      = load_and_merge("other_1",      "num_group1 == 0", ["num_group1"])

# Tax data
tax_a = load_and_merge("tax_registry_a_1", "num_group1 == 0", ["num_group1"])
tax_b = load_and_merge("tax_registry_b_1", "num_group1 == 0", ["num_group1"])
tax_c = load_and_merge("tax_registry_c_1", "num_group1 == 0", ["num_group1"])

# Person data
person_1 = pd.read_parquet(base_path + 'person_1.parquet')
person_2 = pd.read_parquet(base_path + 'person_2.parquet')
person_1 = person_1[person_1["num_group1"]==0].drop(columns="num_group1")
person_2 = person_2[(person_2["num_group1"]==0) & (person_2["num_group2"]==0)].drop(columns=["num_group1", "num_group2"])
person = pd.merge(pd.merge(base_cba, person_1, on="case_id", how="left"), person_2, on="case_id", how="left")
person.drop(columns=[
    "childnum_185L", "relationshiptoclient_642T", "remitter_829L",
    "relationshiptoclient_415T", "housingtype_772L", "birthdate_87D",
    "persontype_792L", "persontype_1072L", "personindex_1023L"
], inplace=True)

# Application data
applprev1 = pd.concat([
    pd.read_parquet(base_path + 'applprev_1_0.parquet'),
    pd.read_parquet(base_path + 'applprev_1_1.parquet')
])
applprev2 = pd.read_parquet(base_path + 'applprev_2.parquet')
applprev1 = applprev1[applprev1["num_group1"]==0].drop(columns="num_group1")
applprev2 = applprev2[(applprev2["num_group1"]==0) & (applprev2["num_group2"]==0)].drop(columns=["num_group1", "num_group2"])
applprev = pd.merge(pd.merge(base_cba, applprev1, on="case_id", how="left"), applprev2, on="case_id", how="left")

# Static data
static = pd.concat([
    pd.read_parquet(base_path + 'static_0_0.parquet'),
    pd.read_parquet(base_path + 'static_0_1.parquet')
])
static = pd.merge(base_cba, static, on="case_id", how="left")

# Credit bureau A
cb_a_parts = []
for i in range(4):
    df = pd.read_parquet(base_path + f'credit_bureau_a_1_{i}.parquet')
    df = df[df["num_group1"] == 0].drop(columns="num_group1")
    cb_a_parts.append(df)
credit_bureau_a_1 = pd.concat(cb_a_parts)

cb_a_parts = []
for i in range(11):
    df = pd.read_parquet(base_path + f'credit_bureau_a_2_{i}.parquet')
    df = df[(df["num_group1"] == 0) & (df["num_group2"] == 0)].drop(columns=["num_group1", "num_group2"])
    cb_a_parts.append(df)
credit_bureau_a_2 = pd.concat(cb_a_parts)

cb = pd.merge(pd.merge(base_cba, credit_bureau_a_1, on="case_id", how="left"),
              credit_bureau_a_2, on="case_id", how="left")

# Credit bureau B
cb_b_1 = pd.read_parquet(base_path + 'credit_bureau_b_1.parquet')
cb_b_2 = pd.read_parquet(base_path + 'credit_bureau_b_2.parquet')
cb_b_1 = cb_b_1[cb_b_1["num_group1"]==0]
cb_b_1.drop(columns=["num_group1"],inplace=True)
cb_b_2 = cb_b_2[(cb_b_2["num_group1"]==0) & (cb_b_2["num_group2"]==0)]
cb_b_2.drop(columns=["num_group1","num_group2"],inplace=True)
cb2 = pd.merge(pd.merge(base_cba, cb_b_1, on="case_id", how="left"),
               cb_b_2, on="case_id", how="left")
static_cb = pd.read_parquet(base_path + 'static_cb_0.parquet')
static_cb = pd.merge(base_cba, static_cb, on="case_id", how="left")

drop_cols = ['WEEK_NUM', 'target', 'date_decision']
dataframes = [person, applprev, static_cb, static, tax_a, tax_b, tax_c, deposit_1, debitcard_1, other_1, cb2]
for df in dataframes:
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

dataframes = [downcast_numeric(convert_columns(df)) for df in dataframes]
cb = downcast_numeric(convert_columns(cb))

merged = cb
for df in dataframes:
    merged = merged.merge(df, on="case_id", how="inner")
    del df
    gc.collect()

del cb
gc.collect()

merged["birth_259D"] = pd.to_datetime(merged["birth_259D"], errors='coerce')
merged["age_at_decision"] = (merged["date_decision"] - merged["birth_259D"]).dt.days // 365
merged.drop(columns=["birth_259D"], inplace=True)

merged = convert_dates_to_days_from_decision(merged, decision_col="date_decision")
merged.drop(columns=["date_decision"], inplace=True)

for col in merged.select_dtypes(include='bool'):
    merged[col] = merged[col].astype('category')


for col in merged.select_dtypes(include='category').columns:
    merged[col] = merged[col].astype(str)

missing_threshold = 0
missing_cols = [col for col in merged.columns if merged[col].isna().mean() > missing_threshold]

flags = pd.DataFrame(
    {f"{col}_missing": merged[col].isna().astype('uint8') for col in missing_cols},
    index=merged.index
)

merged = pd.concat([merged, flags], axis=1)
del flags
gc.collect()

merged.to_parquet('/rds/general/user/ms2524/home/MSc-project/notebooks/oldmerged.parquet', index=False)


