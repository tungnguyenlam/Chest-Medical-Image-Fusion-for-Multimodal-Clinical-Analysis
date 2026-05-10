import pandas as pd
import os
import sys

if not os.path.isdir('data') or not os.path.isdir('camchex'):
    sys.exit("Run from project root: python camchex/data/03_filter_existing_images.py")

# Reads 02_*.csv from data/data-camchex/, filters to rows whose image exists on disk,
# and writes one set of 03_*.csv per image source. The training config picks which set
# (and the matching base dir) to use.
#
# CSV `path` column looks like: images/p11/p11941242/s50000014/<dicom>.jpg
# We strip the leading "images/" and join against each source's base directory.

DATA_CAMCHEX_ROOT = 'data/data-camchex'
OUT_DIR = DATA_CAMCHEX_ROOT

# (suffix, base_dir_for_images) — base_dir is what `images/...` resolves against
SOURCES = [
    ('mimic',  'data/MIMIC-CXR-JPG/files'),
    ('kaggle', 'data/data-kaggle/official_data_iccv_final/files'),
]

SPLITS = ['train', 'development', 'test']

os.makedirs(OUT_DIR, exist_ok=True)


def strip_images_prefix(p: str) -> str:
    # CSV paths are stored as "images/pXX/..."; strip the "images/" prefix so we can
    # join against each source's base directory directly.
    return p[len('images/'):] if p.startswith('images/') else p


for source_tag, base_dir in SOURCES:
    if not os.path.isdir(base_dir):
        print(f"[{source_tag}] Base dir missing: {base_dir} — skipping this source.")
        continue

    print(f"=== Source: {source_tag} (base: {base_dir}) ===")
    for split in SPLITS:
        in_path = os.path.join(DATA_CAMCHEX_ROOT, f'02_{split}.csv')
        out_path = os.path.join(OUT_DIR, f'03_{source_tag}_{split}.csv')

        if not os.path.exists(in_path):
            print(f"  Not found, skipping: {in_path}")
            continue

        print(f"  Filtering {os.path.basename(in_path)} -> {os.path.basename(out_path)}")
        df = pd.read_csv(in_path, low_memory=False)

        rels = df['path'].map(strip_images_prefix)
        existing = rels.map(lambda r: os.path.exists(os.path.join(base_dir, r)))
        filtered_df = df[existing]

        kept, total = len(filtered_df), len(df)
        print(f"    kept {kept} / {total} ({total - kept} missing dropped)")

        filtered_df.to_csv(out_path, index=False)
        print(f"    saved {out_path}")
