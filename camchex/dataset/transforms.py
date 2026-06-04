import cv2
import albumentations as A

def get_transforms(size):
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
        A.Resize(size,size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(),
    ])

    transforms_val = A.Compose([
        A.Resize(size,size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize()
    ])
    return transforms_train, transforms_val

