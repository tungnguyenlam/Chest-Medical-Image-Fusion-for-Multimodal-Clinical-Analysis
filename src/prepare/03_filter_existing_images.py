import argparse
import pandas as pd
import os
import sys
from tqdm import tqdm

tqdm.pandas()

if not os.path.isdir('data') or not os.path.isdir('src'):
    sys.exit("Run from project root: python src/prepare/03_filter_existing_images.py")

_parser = argparse.ArgumentParser(description="Filter merged CSVs to images that exist on disk.")
_parser.add_argument(
    '--subset', default='subset', choices=['full', 'subset', 'kaggle'],
    help="Which image source to filter against (and rewrite path columns for). "
         "'full' = data/MIMIC-CXR-JPG/files. "
         "'subset' = data/subset/MIMIC-CXR-JPG/files. "
         "'kaggle' = data/data-kaggle/official_data_iccv_final/files. "
         "Default: subset."
)
_args, _ = _parser.parse_known_args()

# Reads 02_*.csv from data/data-camchex/, filters to rows whose image exists on disk,
# and writes one set of 03_*.csv per image source. The training config picks which set
# (and the matching base dir) to use.
#
# CSV `path` column looks like: images/p11/p11941242/s50000014/<dicom>.jpg
# We strip the leading "images/" and join against each source's base directory both
# to test existence AND to rewrite the saved `path` column. Training cwd is camchex/,
# so rewritten paths use a `../` prefix to escape into the project root before
# joining with the source base dir. This makes the dataset's open() succeed without
# any per-machine `camchex/images/` symlink.

DATA_CAMCHEX_ROOT = 'data/data-camchex'
OUT_DIR = DATA_CAMCHEX_ROOT

# (suffix, base_dir_for_images, base_dir_from_camchex_cwd)
#   base_dir_for_images       — used at filter time (this script runs from project root)
#   base_dir_from_camchex_cwd — written into the CSV (training script runs from camchex/)
# (output_tag, base_dir_at_project_root, base_dir_from_camchex_cwd)
_SOURCE_BY_SUBSET = {
    'full':         ('mimic',  'data/MIMIC-CXR-JPG/files',
                                '../data/MIMIC-CXR-JPG/files'),
    'subset': ('mimic',  'data/subset/MIMIC-CXR-JPG/files',
                                '../data/subset/MIMIC-CXR-JPG/files'),
    'kaggle':       ('kaggle', 'data/data-kaggle/official_data_iccv_final/files',
                                '../data/data-kaggle/official_data_iccv_final/files'),
}
SOURCES = [_SOURCE_BY_SUBSET[_args.subset]]
print(f"[03_filter_existing_images] subset={_args.subset}  source={SOURCES[0][0]}  base={SOURCES[0][1]}")

SPLITS = ['train', 'development', 'test']

os.makedirs(OUT_DIR, exist_ok=True)


def strip_images_prefix(p: str) -> str:
    # CSV paths are stored as "images/pXX/..."; strip the "images/" prefix so we can
    # join against each source's base directory directly.
    return p[len('images/'):] if p.startswith('images/') else p


for source_tag, base_dir, base_dir_from_camchex in SOURCES:
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
        existing = rels.progress_map(lambda r: os.path.exists(os.path.join(base_dir, r)))
        filtered_df = df[existing].copy()
        filtered_df['path'] = rels[existing].map(
            lambda r: os.path.join(base_dir_from_camchex, r)
        )

        kept, total = len(filtered_df), len(df)
        print(f"    kept {kept} / {total} ({total - kept} missing dropped)")

        filtered_df.to_csv(out_path, index=False)
        print(f"    saved {out_path}")
