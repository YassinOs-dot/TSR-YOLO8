import cv2
from skimage import transform, exposure
import numpy as np

def image_processing(img):
    """
    Preprocess image for CNN model
    """
    # Convert BGR to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 32x32 (GTSRB standard)
    img = transform.resize(img, (32, 32))
    
    # Histogram equalization
    img = exposure.equalize_adapthist(img, clip_limit=0.1)
    
    # Normalize to [0, 1]
    img = img.astype("float32")
    
    return img
