import hashlib
import os

import cv2
import albumentations as A
import numpy as np
from pathlib import Path

from src.dataloader.image_channel_preprocessing import (
    CHANNEL_MODES,
    CHANNEL_STATS,
    PreprocessConfig,
    build_channels,
)

def get_transforms(size, mode=None):
    """Build (train, val) albumentations pipelines.

    When ``mode`` is given, images are deterministic 3-channel float [0, 1]
    arrays from :func:`build_channels`, so normalization uses the precomputed
    per-channel stats with ``max_pixel_value=1.0``. When ``mode`` is ``None`` the
    legacy ImageNet uint8 normalization is preserved unchanged.
    """
    if mode:
        stats = CHANNEL_STATS[mode]
        normalize = A.Normalize(mean=stats["mean"], std=stats["std"], max_pixel_value=1.0)
    else:
        normalize = A.Normalize()

    # RandomResizedCrop already outputs (size, size) and every later aug here is
    # size-preserving, so a trailing A.Resize(size, size) would be a guaranteed
    # no-op (a wasted LANCZOS4 pass on every item). Omitted unconditionally.
    transforms_train = A.Compose([
        A.RandomResizedCrop((size,size), scale=(0.9, 1), p=1, interpolation=cv2.INTER_LANCZOS4),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.0625, 0.0625),
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.OneOf([
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.ElasticTransform(),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(),
            A.GaussianBlur(),
            A.MotionBlur(),
            A.MedianBlur(),
        ], p=0.2),
        normalize,
    ])

    # With a channel ``mode``, build_channels emits arrays already resized to
    # (size, size) -- out_size is data_cfg["size"] and is folded into the cache
    # key -- so the val Resize is a no-op and is dropped. In legacy mode (None)
    # images arrive at native resolution, so the Resize is kept (also the only
    # thing making the batch collatable).
    val_steps = []
    if not mode:
        val_steps.append(A.Resize(size, size, interpolation=cv2.INTER_LANCZOS4))
    val_steps.append(normalize)
    transforms_val = A.Compose(val_steps)
    return transforms_train, transforms_val

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _path_candidates(path):
    raw_path = Path(str(path))
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend([
            Path.cwd() / raw_path,
            _REPO_ROOT / raw_path,
            _REPO_ROOT / "camchex" / raw_path,
        ])

    # Deliberately no .resolve(): canonicalizing lstat()s every path component
    # (~10 extra syscalls per call), which is brutal on WSL /mnt/<drive> drvfs
    # mounts. cv2.imread works fine with non-canonical paths, and image_cache_path
    # canonicalizes its own cache key independently, so keys stay stable.
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def _first_existing(path):
    for candidate in _path_candidates(path):
        if candidate.exists():
            return str(candidate)
    return None


def resolve_image_path(path):
    found = _first_existing(path)
    return found if found is not None else str(path)


def resolve_preferred_image_path(path):
    resized = _first_existing(str(path).replace(".jpg", "_resized_1024.jpg"))
    if resized is not None:
        return resized
    return resolve_image_path(path)


def _safe_decode_jpeg(path):
    """Decode a JPEG with cv2.

    Returns an HWC uint8 RGB ndarray, or None if every attempt fails.
    """
    paths = [resolve_image_path(path)]
    if "_resized_1024.jpg" in paths[0]:
        paths.append(paths[0].replace("_resized_1024.jpg", ".jpg"))

    for p in paths:
        # cv2.imread returns None for a missing/unreadable file, so an explicit
        # existence stat would just be a redundant syscall on drvfs.
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def image_cache_path(cache_dir, image_path):
    resolved = str(Path(image_path).resolve(strict=False))
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.npy"


def load_cached_rgb(cache_dir, image_path):
    if not cache_dir:
        return None
    path = image_cache_path(cache_dir, image_path)
    if not path.exists():
        return None
    return np.load(path)


def make_preprocess_config(cfg):
    """Build a PreprocessConfig from a datamodule cfg (out_size follows ``size``)."""
    return PreprocessConfig(out_size=int(cfg.get("size", 512)))


def channel_cache_path(cache_dir, image_path, mode, preprocess_cfg):
    """Cache path keyed by (raw path string, mode, preprocessing fingerprint).

    The key is derived from the path STRING only -- no ``Path.resolve()`` / stat --
    so a cache hit never touches the source filesystem. That matters enormously
    when the images live on a slow mount (e.g. WSL ``/mnt/<drive>`` drvfs): both
    this precompute scan and every training epoch can decide "already cached?"
    without a single cross-boundary stat. Resolution to the real on-disk file
    happens only on a miss, inside :func:`load_or_build_channels`.

    Callers must pass a *stable* path string (the same one each epoch) -- the raw
    ``path``/``img_paths`` value straight from the dataframe/parquet. The
    fingerprint keeps the cache self-invalidating on mode/size/filter changes.
    """
    key = f"{os.path.normpath(str(image_path))}|{mode}|{preprocess_cfg.fingerprint()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return Path(cache_dir) / mode / f"{digest}.npy"


def _atomic_save_uint8(path, arr):
    """Write ``arr`` to ``path`` via a per-process temp file + atomic rename.

    Prevents partial-file reads when multiple dataloader workers populate the
    shared cache concurrently.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def load_or_build_channels(image_path, mode, preprocess_cfg, cache_dir=None):
    """Return an HWC float32 [0, 1] 3-channel array for ``mode`` (or None).

    ``image_path`` is the raw (dataframe) path string. With ``cache_dir`` set:
    cache hit -> load uint8 and de-quantize (no source-FS access); miss -> resolve
    the preferred on-disk file, decode, build channels, quantize to uint8, store
    atomically. The uint8 round-trip is applied on the build path too so
    cached/uncached epochs are numerically identical. Without ``cache_dir`` the
    channels are built fresh each call (no quantization).

    Resolution (stat-heavy on slow mounts) is deferred to the miss path, so warm
    epochs and the precompute scan never stat the source filesystem.
    """
    if mode not in CHANNEL_MODES:
        raise ValueError(f"Unknown channel_mode {mode!r}; expected one of {sorted(CHANNEL_MODES)}")

    if cache_dir:
        cpath = channel_cache_path(cache_dir, image_path, mode, preprocess_cfg)
        if cpath.exists():
            try:
                arr = np.load(cpath)
                return arr.astype(np.float32) / 255.0
            except (ValueError, OSError):
                pass  # corrupt / partially-written file -> rebuild below

    rgb = _safe_decode_jpeg(resolve_preferred_image_path(image_path))
    if rgb is None:
        return None
    channels = build_channels(rgb, mode, preprocess_cfg)  # float32 [0, 1]

    if cache_dir:
        quantized = np.clip(np.round(channels * 255.0), 0, 255).astype(np.uint8)
        _atomic_save_uint8(cpath, quantized)
        return quantized.astype(np.float32) / 255.0
    return channels
