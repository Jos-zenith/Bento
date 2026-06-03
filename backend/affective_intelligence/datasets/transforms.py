"""Image transforms for emotion recognition datasets."""

from torchvision import transforms


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get training transforms with augmentation.
    
    Args:
        img_size: Target image size (default: 224 for EfficientNet-B0)
    
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get validation transforms (minimal augmentation).
    
    Args:
        img_size: Target image size
    
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_inference_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get inference transforms (same as validation).
    
    Args:
        img_size: Target image size
    
    Returns:
        Composed transforms
    """
    return get_val_transforms(img_size)
