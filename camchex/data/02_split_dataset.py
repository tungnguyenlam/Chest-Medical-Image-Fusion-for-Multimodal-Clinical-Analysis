import pandas as pd
import os
import sys

if not os.path.isdir('data') or not os.path.isdir('camchex'):
    sys.exit("Run from project root: python camchex/data/02_split_dataset.py")

DATA_CAMCHEX_ROOT = 'data/data-camchex'
_CXRLT_2023 = 'data/CXR-LT/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0/cxr-lt-2023'

print("Loading 01_merged.csv...")
dataset_df = pd.read_csv(f'{DATA_CAMCHEX_ROOT}/01_merged.csv', low_memory=False)

print("Loading CXR-LT splits...")
train_ids = set(pd.read_csv(f'{_CXRLT_2023}/train.csv', usecols=['dicom_id'])['dicom_id'])
dev_ids = set(pd.read_csv(f'{_CXRLT_2023}/development.csv', usecols=['dicom_id'])['dicom_id'])
test_ids = set(pd.read_csv(f'{_CXRLT_2023}/test.csv', usecols=['dicom_id'])['dicom_id'])

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
