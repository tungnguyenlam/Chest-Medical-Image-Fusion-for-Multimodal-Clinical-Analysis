import cv2
import albumentations as A
import os
import warnings
import jpeg4py as jpeg

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



def _safe_decode_jpeg(path):
    """Decode a JPEG, falling back to cv2 if jpeg4py fails (e.g. truncated file).

    Returns an HWC uint8 RGB ndarray, or None if every attempt fails.
    """
    candidates = [path]
    if "_resized_1024.jpg" in path:
        candidates.append(path.replace("_resized_1024.jpg", ".jpg"))
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            return jpeg.JPEG(p).decode()
        except Exception as e:
            warnings.warn(f"jpeg4py failed on {p}: {e}; falling back to cv2")
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        warnings.warn(f"cv2 also failed to decode {p}")
    return None
