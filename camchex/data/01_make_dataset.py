import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

if not os.path.isdir('data') or not os.path.isdir('camchex'):
    sys.exit("Run from project root: python camchex/data/01_make_dataset.py")

sys.path.append('mimic-cxr/txt')
import section_parser as sp

_parser = argparse.ArgumentParser(description="Build the camchex merged dataset.")
_parser.add_argument(
    '--subset', default='seed42_10pct', choices=['full', 'seed42_10pct'],
    help="Which MIMIC-CXR(-JPG) source to use. 'full' reads from data/MIMIC-CXR* (school server); "
         "'seed42_10pct' reads from data/subset/MIMIC-CXR* (cloud GPU). Default: seed42_10pct."
)
_args, _ = _parser.parse_known_args()

DATA_ROOT = 'data'
DATA_CAMCHEX_ROOT = 'data/data-camchex'
MIMIC_ROOT = 'data' if _args.subset == 'full' else 'data/subset'
print(f"[01_make_dataset] subset={_args.subset}  MIMIC_ROOT={MIMIC_ROOT}")

mimic_cxr_metadata_fp     = f'{MIMIC_ROOT}/MIMIC-CXR-JPG/mimic-cxr-2.0.0-metadata.csv'
mimic_cxr_split_fp        = f'{MIMIC_ROOT}/MIMIC-CXR-JPG/mimic-cxr-2.0.0-split.csv'
mimic_iv_ed_triage_fp     = f'{DATA_ROOT}/MIMIC-IV-ED-2-2/mimic-iv-ed-2.2/ed/triage.csv.gz'
mimic_iv_ed_edstays_fp    = f'{DATA_ROOT}/MIMIC-IV-ED-2-2/mimic-iv-ed-2.2/ed/edstays.csv.gz'
mimic_iv_ed_vitalsigns_fp = f'{DATA_ROOT}/MIMIC-IV-ED-2-2/mimic-iv-ed-2.2/ed/vitalsign.csv.gz'
reports_base_path         = f'{MIMIC_ROOT}/MIMIC-CXR/files'

_CXRLT_2023        = f'{DATA_ROOT}/CXR-LT/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0/cxr-lt-2023'
labels_train_fp    = f'{_CXRLT_2023}/train.csv'
labels_validate_fp = f'{_CXRLT_2023}/development.csv'
labels_test_fp     = f'{_CXRLT_2023}/test.csv'

# --- Config ---
CPU_FRACTION = 0.5  # Use 50% of CPU cores by default to avoid server termination

# Output: step 1 merged dataset (read by 02_split_dataset.py)
output_fp = f'{DATA_CAMCHEX_ROOT}/01_merged.csv'

os.makedirs(DATA_CAMCHEX_ROOT, exist_ok=True)


def _clean_report(report):
    report = report.replace('\\n', ' ').replace('\n', ' ')
    report = re.sub(r'(?<![a-zA-Z])/{1,}|/{1,}(?![a-zA-Z])', '', report)
    report = re.sub(r'\s+', ' ', report).strip()
    report_cleaner = lambda t: t.replace('__', '_') \
        .replace('..', '.').replace('  ', ' ') \
        .replace('1. ', '').replace('2. ', '').replace('3. ', '') \
        .replace('4. ', '').replace('5. ', '') \
        .strip().lower()
    sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '')
                                    .replace("'", '').strip().lower())
    cleaned_sentences = [sent_cleaner(sent) for sent in report_cleaner(report).split('. ') if sent]
    cleaned_report = ' . '.join(cleaned_sentences).strip() + ' .'
    return re.sub(r'\s+', ' ', cleaned_report).strip()


# Module-level globals for multiprocessing workers (set before Pool creation)
custom_section_names = {}
custom_indices = {}


def _parse_single_report(args):
    """Parse a single report file. Used by multiprocessing.Pool workers."""
    idx, subject_id, study_id = args
    subject_id_str = str(subject_id)
    study_id_str = f"s{study_id}"
    report_path = os.path.join(
        reports_base_path, f"p{subject_id_str[:2]}",
        f"p{subject_id_str}", f"{study_id_str}.txt"
    )

    if not os.path.exists(report_path):
        return None

    with open(report_path, 'r') as f:
        text = f.read()

    if study_id_str in custom_indices:
        ci = custom_indices[study_id_str]
        text = text[ci[0]:ci[1]]

    sections, section_names, _ = sp.section_text(text)
    section_dict = dict(zip(section_names, sections))

    result = {'_idx': idx}

    if study_id_str in custom_section_names:
        custom_sec = custom_section_names[study_id_str]
        if custom_sec in section_dict:
            result[custom_sec] = _clean_report(section_dict[custom_sec])
        return result

    for section in ['impression', 'findings', 'last_paragraph', 'comparison', 'indication', 'history']:
        if section in section_dict:
            result[section] = _clean_report(section_dict[section])

    return result


# --- Load image metadata and ED data ---
print("Loading image metadata...")
metadata_df = pd.read_csv(mimic_cxr_metadata_fp)
cxr_df = metadata_df.drop(columns=['dicom_id', 'ViewPosition', 'Rows', 'Columns']).drop_duplicates(subset='study_id').reset_index(drop=True)

cxr_df = cxr_df.sort_values(by=['subject_id', 'StudyDate']).reset_index(drop=True)
cxr_df['PreviousStudy'] = cxr_df.groupby('subject_id')['study_id'].shift(1).astype('Int64')
cxr_df['StudyDateTime'] = pd.to_datetime((cxr_df['StudyDate'] * 1000000) + cxr_df['StudyTime'].astype(int), format="%Y%m%d%H%M%S")
cxr_df = cxr_df.drop(columns=['StudyDate', 'StudyTime'])

print("Loading ED data...")
triage_df = pd.read_csv(mimic_iv_ed_triage_fp)
edstays_df = pd.read_csv(mimic_iv_ed_edstays_fp)
ed_df = pd.merge(triage_df, edstays_df, on=['subject_id', 'stay_id'])
ed_df['intime'] = pd.to_datetime(ed_df['intime'])
ed_df['outtime'] = pd.to_datetime(ed_df['outtime'])

mimic_df = pd.merge(cxr_df, ed_df, on='subject_id', how='left')
mimic_df['time_to_intime'] = (mimic_df['StudyDateTime'] - mimic_df['intime']).abs().dt.total_seconds()
mimic_df['time_to_outtime'] = (mimic_df['StudyDateTime'] - mimic_df['outtime']).abs().dt.total_seconds()

closest_stays_idx = mimic_df.dropna(subset=['time_to_intime']).groupby('study_id')['time_to_intime'].idxmin()
mimic_df = pd.concat([
    mimic_df.loc[closest_stays_idx],
    mimic_df[~mimic_df['study_id'].isin(mimic_df.loc[closest_stays_idx, 'study_id'])]
]).drop(columns=['time_to_intime', 'time_to_outtime']).reset_index(drop=True)

print("Loading vital signs...")
vitalsigns_df = pd.read_csv(mimic_iv_ed_vitalsigns_fp)
vitalsigns_df['charttime'] = pd.to_datetime(vitalsigns_df['charttime'])
merged_df = pd.merge(mimic_df, vitalsigns_df, on=['subject_id', 'stay_id'], suffixes=('', '_chart'), how='left')
merged_df['time_diff'] = (merged_df['StudyDateTime'] - merged_df['charttime']).abs()

vital_signs = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
closest_charts = (
    merged_df[merged_df[vital_signs].isna().any(axis=1)]
    .sort_values(by=['study_id', 'time_diff'])
    .drop_duplicates(subset=['study_id'], keep='first')
)
for vital in vital_signs:
    mimic_df[vital] = mimic_df[vital].combine_first(closest_charts[f"{vital}_chart"])
mimic_df = mimic_df.reset_index(drop=True)

print("Loading CXR-LT labels...")
label_cols = [
    "study_id", "Atelectasis", "Calcification of the Aorta", "Cardiomegaly", "Consolidation", "Edema", "Emphysema",
    "Enlarged Cardiomediastinum", "Fibrosis", "Fracture", "Hernia", "Infiltration", "Lung Lesion", "Lung Opacity",
    "Mass", "No Finding", "Nodule", "Pleural Effusion", "Pleural Other", "Pleural Thickening", "Pneumomediastinum",
    "Pneumonia", "Pneumoperitoneum", "Pneumothorax", "Subcutaneous Emphysema", "Support Devices", "Tortuous Aorta"
]
df_train = pd.read_csv(labels_train_fp, usecols=label_cols)
df_val   = pd.read_csv(labels_validate_fp, usecols=label_cols)
df_test  = pd.read_csv(labels_test_fp, usecols=label_cols)
df_train['split'] = 'train'
df_val['split']   = 'validate'
df_test['split']  = 'test'
labels_df = pd.concat([df_train, df_val, df_test]).drop_duplicates(subset="study_id")
cxr_df = cxr_df.merge(labels_df, on="study_id", how="left")

# --- Parse text reports (parallel) ---
print("Parsing reports (missing reports are skipped)...")
custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

for col in ['impression', 'findings', 'last_paragraph', 'comparison', 'indication', 'history']:
    mimic_df[col] = None

# Build lightweight argument list: (df_index, subject_id, study_id)
report_args = list(zip(mimic_df.index, mimic_df['subject_id'], mimic_df['study_id']))

n_workers = max(1, int(cpu_count() * CPU_FRACTION))
print(f"Using {n_workers} parallel workers for report parsing...")

mp_ctx = __import__('multiprocessing').get_context('fork')
with mp_ctx.Pool(n_workers) as pool:
    results = list(tqdm(
        pool.imap_unordered(_parse_single_report, report_args, chunksize=256),
        total=len(report_args),
        desc="Processing reports"
    ))

# Apply parsed results back to DataFrame
for result in results:
    if result is None:
        continue
    idx = result.pop('_idx')
    for col, val in result.items():
        mimic_df.at[idx, col] = val

print("Saving progress checkpoint...")
mimic_df.to_csv(f'{DATA_CAMCHEX_ROOT}/01_progress.csv', index=False)

# --- Build report and clinical indication fields ---
def choose_report(row):
    for col in ['findings', 'impression', 'last_paragraph']:
        if pd.notna(row[col]) and str(row[col]).strip():
            return row[col]
    return np.nan

mimic_df['report'] = mimic_df.apply(choose_report, axis=1)

for study_id_str, section_name in custom_section_names.items():
    sid = int(study_id_str[1:])
    if section_name in mimic_df.columns:
        mimic_df.loc[mimic_df['study_id'] == sid, 'report'] = mimic_df.loc[mimic_df['study_id'] == sid, section_name]
        mimic_df.loc[mimic_df['study_id'] == sid, section_name] = np.nan

def merge_clinical_indication(row):
    ind  = str(row['indication']) if pd.notna(row['indication']) else ''
    hist = str(row['history'])    if pd.notna(row['history'])    else ''
    combined = (ind.strip() + (' ' if ind.strip() and hist.strip() else '') + hist.strip()).strip()
    return combined if combined else np.nan

mimic_df['ClinicalIndication'] = mimic_df.apply(merge_clinical_indication, axis=1)
mimic_df.drop(columns=['impression', 'findings', 'last_paragraph', 'comparison', 'indication',
                        'examination', 'technique', 'recommendations', 'history', 'note'],
              inplace=True, errors='ignore')

# --- Merge dicom-level metadata and labels ---
print("Merging dicom metadata...")
meta_df = metadata_df[['dicom_id', 'study_id', 'ViewPosition']]
mimic_df.drop(columns=['PerformedProcedureStepDescription', 'ProcedureCodeSequence_CodeMeaning',
                        'ViewCodeSequence_CodeMeaning', 'PatientOrientationCodeSequence_CodeMeaning'],
              inplace=True, errors='ignore')

merged_df = mimic_df.merge(meta_df[['study_id', 'dicom_id', 'ViewPosition']], on='study_id', how='left')
merged_df = merged_df.dropna(subset=['report'])
merged_df = merged_df.rename(columns={'ClinicalIndication': 'clinical_indication'})

print("Merging dicom-level CXR-LT labels...")
labels_train_df    = pd.read_csv(labels_train_fp)
labels_validate_df = pd.read_csv(labels_validate_fp)
labels_test_df     = pd.read_csv(labels_test_fp)
labels_train_df['split']    = 'train'
labels_validate_df['split'] = 'validate'
labels_test_df['split']     = 'test'
labels_df = pd.concat([labels_train_df, labels_validate_df, labels_test_df])
labels_df.drop(columns=['subject_id', 'study_id', 'ViewPosition',
                         'ViewCodeSequence_CodeMeaning', 'path'], inplace=True, errors='ignore')

merged_df = merged_df.merge(labels_df, on='dicom_id', how='left')

print(f"Saving {output_fp}...")
merged_df.to_csv(output_fp, index=False)

# --- Split into train / development / test ---
merged_df['path'] = merged_df.apply(
    lambda row: f"images/p{str(row['subject_id'])[:2]}/p{str(row['subject_id'])}/s{row['study_id']}/{row['dicom_id']}.jpg",
    axis=1
)

train    = merged_df[merged_df['split'] == 'train'].drop(columns=['split'])
validate = merged_df[merged_df['split'] == 'validate'].drop(columns=['split'])
test     = merged_df[merged_df['split'] == 'test'].drop(columns=['split'])

train.to_csv(f'{DATA_CAMCHEX_ROOT}/02_train.csv', index=False)
validate.to_csv(f'{DATA_CAMCHEX_ROOT}/02_development.csv', index=False)
test.to_csv(f'{DATA_CAMCHEX_ROOT}/02_test.csv', index=False)

print(f"Done. Outputs in {DATA_CAMCHEX_ROOT}/")
print(f"  01_merged.csv       ({len(merged_df)} rows)")
print(f"  02_train.csv        ({len(train)} rows)")
print(f"  02_development.csv  ({len(validate)} rows)")
print(f"  02_test.csv         ({len(test)} rows)")
