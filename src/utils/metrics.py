import numpy as np

def normalise_rich_images(images, max_val=8):
    """
    Normalise RICH detector images from integer counts to the [-1, 1] range.
    
    Args:
        images: numpy array of images with integer values representing hit counts
        max_val: maximum expected value (used for scaling)
    
    Returns:
        Normalised images in the [-1, 1] range
    """
    # First, clip values to the expected range
    images = np.clip(images, 0, max_val)
    
    # Scale from [0, max_val] to [0, 2]
    images = images * (2 / max_val)
    
    # Shift from [0, 2] to [-1, 1]
    images = images - 1
    
    return images

def denormalise_rich_images(normalised_images, max_val=8):
    """
    Convert images from [-1, 1] range back to integer counts.
    
    Args:
        normalised_images: numpy array of images in [-1, 1] range
        max_val: maximum expected value (used for scaling)
    
    Returns:
        Images with integer values representing hit counts
    """
    # Shift from [-1, 1] to [0, 2]
    images = normalised_images + 1
    
    # Scale from [0, 2] to [0, max_val]
    images = images * (max_val / 2)
    
    # Round to nearest integer and clip
    images = np.clip(np.round(images), 0, max_val).astype(np.int32)
    
    return images