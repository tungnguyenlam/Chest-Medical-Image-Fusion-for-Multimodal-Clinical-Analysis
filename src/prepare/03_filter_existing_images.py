import argparse
import functools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image
from tqdm import tqdm

# CXRs are legitimately large; don't let Pillow's decompression-bomb guard flag them
# as corrupt. We deliberately do NOT enable LOAD_TRUNCATED_IMAGES: a truncated JPEG
# must raise here so we can filter it, instead of being silently half-decoded the way
# cv2.imread (the training decode path) would — that half-decode + libjpeg warning spam
# is exactly the slowdown this step exists to prevent.
Image.MAX_IMAGE_PIXELS = None

if not os.path.isdir('data') or not os.path.isdir('src'):
    sys.exit("Run from project root: python src/prepare/03_filter_existing_images.py")

_parser = argparse.ArgumentParser(description="Filter merged CSVs to images that exist (and decode) on disk.")
_parser.add_argument(
    '--subset', default='subset', choices=['full', 'subset', 'kaggle'],
    help="Which image source to filter against (and rewrite path columns for). "
         "'full' = data/MIMIC-CXR-JPG/files. "
         "'subset' = data/subset/MIMIC-CXR-JPG/files. "
         "'kaggle' = data/data-kaggle/official_data_iccv_final/files. "
         "Default: subset."
)
_parser.add_argument(
    '--verify-images', action=argparse.BooleanOptionalAction, default=True,
    help="Fully decode each JPEG to drop corrupt/truncated files (not just missing ones). "
         "On (default) makes the pass slower but stops corrupt images from spamming "
         "libjpeg warnings and slowing training/precompute later. Use --no-verify-images "
         "for a fast existence-only filter."
)
_parser.add_argument(
    '--workers', type=int, default=min(8, os.cpu_count() or 1),
    help="Parallel workers for the existence/decode check. Default: min(8, cpu_count)."
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
print(f"[03_filter_existing_images] subset={_args.subset}  source={SOURCES[0][0]}  base={SOURCES[0][1]}  "
      f"verify_images={_args.verify_images}  workers={_args.workers}")

SPLITS = ['train', 'development', 'test']

os.makedirs(OUT_DIR, exist_ok=True)


def strip_images_prefix(p: str) -> str:
    # CSV paths are stored as "images/pXX/..."; strip the "images/" prefix so we can
    # join against each source's base directory directly.
    return p[len('images/'):] if p.startswith('images/') else p


def _check_one(abs_path: str, verify: bool) -> str:
    """Classify a single image path as 'ok', 'missing', or 'corrupt'.

    With verify=True we open and fully decode the JPEG (load()), so a truncated or
    otherwise corrupt file raises and is reported 'corrupt'. Pillow decodes strictly
    here (truncation tolerance left off on purpose), catching the "premature end of
    data segment" files that cv2.imread would only half-decode at training time.
    """
    if not verify:
        return 'ok' if os.path.exists(abs_path) else 'missing'
    try:
        with Image.open(abs_path) as im:
            im.load()
        return 'ok'
    except FileNotFoundError:
        return 'missing'
    except Exception:
        return 'corrupt'


def classify_images(abs_paths: list[str], verify: bool, workers: int, desc: str) -> list[str]:
    """Run _check_one over all paths in parallel, preserving input order."""
    fn = functools.partial(_check_one, verify=verify)
    if workers <= 1 or len(abs_paths) <= 1:
        return [fn(p) for p in tqdm(abs_paths, desc=desc)]
    # Decode is CPU-bound; chunk the work so per-task IPC overhead stays small.
    chunksize = max(1, min(256, len(abs_paths) // (workers * 4) or 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(fn, abs_paths, chunksize=chunksize),
                         total=len(abs_paths), desc=desc))


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
        abs_paths = [os.path.join(base_dir, r) for r in rels]
        statuses = pd.Series(
            classify_images(abs_paths, _args.verify_images, _args.workers,
                            desc=f"{source_tag}/{split}"),
            index=df.index,
        )

        keep_mask = statuses == 'ok'
        filtered_df = df[keep_mask].copy()
        filtered_df['path'] = rels[keep_mask].map(
            lambda r: os.path.join(base_dir_from_camchex, r)
        )

        kept, total = len(filtered_df), len(df)
        n_missing = int((statuses == 'missing').sum())
        n_corrupt = int((statuses == 'corrupt').sum())
        print(f"    kept {kept} / {total} ({n_missing} missing, {n_corrupt} corrupt dropped)")

        # Record corrupt paths so they can be inspected / re-fetched, not just silently dropped.
        if n_corrupt:
            corrupt_path = os.path.join(OUT_DIR, f'03_{source_tag}_{split}_corrupt.txt')
            corrupt_rels = rels[statuses == 'corrupt']
            with open(corrupt_path, 'w') as f:
                f.write('\n'.join(corrupt_rels) + '\n')
            print(f"    wrote {n_corrupt} corrupt paths to {corrupt_path}")

        filtered_df.to_csv(out_path, index=False)
        print(f"    saved {out_path}")
