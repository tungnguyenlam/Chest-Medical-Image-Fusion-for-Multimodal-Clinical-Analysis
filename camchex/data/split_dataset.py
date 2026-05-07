import pandas as pd
import os

DATA_ROOT = 'data'
_CXRLT_2023 = f'{DATA_ROOT}/CXR-LT/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0/cxr-lt-2023'

print("Loading dataset.csv...")
dataset_df = pd.read_csv(f'{DATA_ROOT}/dataset.csv', low_memory=False)

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

print("Adding 'path' column...")
for df in [train_df, dev_df, test_df]:
    df['path'] = df.apply(
        lambda row: f"images/p{str(row['subject_id'])[:2]}/p{str(row['subject_id'])}/s{row['study_id']}/{row['dicom_id']}.jpg",
        axis=1
    )

os.makedirs('camchex/data', exist_ok=True)
train_df.to_csv('camchex/data/train.csv', index=False)
dev_df.to_csv('camchex/data/development.csv', index=False)
test_df.to_csv('camchex/data/test.csv', index=False)

print("Saved successfully to camchex/data/!")
