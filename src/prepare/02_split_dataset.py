import pandas as pd
import os
import sys

if not os.path.isdir('data') or not os.path.isdir('src'):
    sys.exit("Run from project root: python src/prepare/02_split_dataset.py")

sys.path.insert(0, os.getcwd())
from src.dataloader.cxr_lt import load_cxr_lt_labels

DATA_CAMCHEX_ROOT = 'data/data-camchex'
CXRLT_VERSION = 'cxr-lt-2023'

print("Loading 01_merged.csv...")
dataset_df = pd.read_csv(f'{DATA_CAMCHEX_ROOT}/01_merged.csv', low_memory=False)

print("Loading CXR-LT splits...")
# Split assignment comes from the shared helper (normalized split column), so the
# 2023 release layout lives in one place. Stage 04 re-splits for the 2024 release.
_cxrlt_df, _, _ = load_cxr_lt_labels('data', version=CXRLT_VERSION)
train_ids = set(_cxrlt_df.loc[_cxrlt_df['split'] == 'train', 'dicom_id'])
dev_ids = set(_cxrlt_df.loc[_cxrlt_df['split'] == 'validate', 'dicom_id'])
test_ids = set(_cxrlt_df.loc[_cxrlt_df['split'] == 'test', 'dicom_id'])

print("Creating splits...")
train_df = dataset_df[dataset_df['dicom_id'].isin(train_ids)].copy()
dev_df = dataset_df[dataset_df['dicom_id'].isin(dev_ids)].copy()
test_df = dataset_df[dataset_df['dicom_id'].isin(test_ids)].copy()

print(f"Train size: {len(train_df)}")
print(f"Dev size: {len(dev_df)}")
print(f"Test size: {len(test_df)}")

for df in [train_df, dev_df, test_df]:
    df['path'] = df.apply(
        lambda row: f"images/p{str(row['subject_id'])[:2]}/p{str(row['subject_id'])}/s{row['study_id']}/{row['dicom_id']}.jpg",
        axis=1
    )

train_df.to_csv(f'{DATA_CAMCHEX_ROOT}/02_train.csv', index=False)
dev_df.to_csv(f'{DATA_CAMCHEX_ROOT}/02_development.csv', index=False)
test_df.to_csv(f'{DATA_CAMCHEX_ROOT}/02_test.csv', index=False)

print(f"Saved 02_train.csv, 02_development.csv, 02_test.csv to {DATA_CAMCHEX_ROOT}/")
