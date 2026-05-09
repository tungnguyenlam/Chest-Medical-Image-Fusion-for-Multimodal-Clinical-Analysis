import pandas as pd
import os
import sys

if not os.path.isdir('data') or not os.path.isdir('camchex'):
    sys.exit("Run from project root: python camchex/data/03_filter_existing_images.py")

# Run from project root: python camchex/data/03_filter_existing_images.py
# Reads 02_*.csv from data/data-camchex/, filters to rows where the image exists
# under camchex/images/ (the directory training uses), writes 03_*.csv to camchex/data/

DATA_CAMCHEX_ROOT = 'data/data-camchex'
IMAGE_BASE = 'camchex'
OUT_DIR = 'camchex/data'

splits = [
    ('02_train.csv',       '03_train.csv'),
    ('02_development.csv', '03_development.csv'),
    ('02_test.csv',        '03_test.csv'),
]

os.makedirs(OUT_DIR, exist_ok=True)

for in_name, out_name in splits:
    in_path = os.path.join(DATA_CAMCHEX_ROOT, in_name)
    if not os.path.exists(in_path):
        print(f"Not found, skipping: {in_path}")
        continue

    print(f"Filtering {in_name}...")
    df = pd.read_csv(in_path)

    existing = df['path'].apply(lambda p: os.path.exists(os.path.join(IMAGE_BASE, p)))
    filtered_df = df[existing]
    print(f"  -> Kept {len(filtered_df)} / {len(df)} rows ({len(df) - len(filtered_df)} missing images dropped)")

    out_path = os.path.join(OUT_DIR, out_name)
    filtered_df.to_csv(out_path, index=False)
    print(f"  -> Saved to {out_path}")
