import cv2
import albumentations as A
from pathlib import Path

def get_transforms(size):
    transforms_train = A.Compose([
        A.RandomResizedCrop((size,size), scale=(0.9, 1), p=1, interpolation=cv2.INTER_LANCZOS4), 
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
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
        A.Resize(size,size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(),
    ])

    transforms_val = A.Compose([
        A.Resize(size,size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize()
    ])
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

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def resolve_image_path(path):
    for candidate in _path_candidates(path):
        if candidate.exists():
            return str(candidate)
    return str(path)


def resolve_preferred_image_path(path):
    resized = str(path).replace(".jpg", "_resized_1024.jpg")
    resized_path = resolve_image_path(resized)
    if Path(resized_path).exists():
        return resized_path
    return resolve_image_path(path)


def _safe_decode_jpeg(path):
    """Decode a JPEG with cv2.

    Returns an HWC uint8 RGB ndarray, or None if every attempt fails.
    """
    paths = [resolve_image_path(path)]
    if "_resized_1024.jpg" in paths[0]:
        paths.append(resolve_image_path(paths[0].replace("_resized_1024.jpg", ".jpg")))

    for p in paths:
        if not Path(p).exists():
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None
