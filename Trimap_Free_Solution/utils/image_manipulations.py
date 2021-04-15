# These functions are copied from 'Pymatting'.

from PIL import Image
import numpy as np

def save_image(path, image):

    assert image.dtype in [np.uint8, np.float32, np.float64]

    if image.dtype in [np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)


    Image.fromarray(image).save(path)

def stack_images(*images):
    
    images = [
        (image if len(image.shape) == 3 else image[:, :, np.newaxis])
        for image in images
    ]
    return np.concatenate(images, axis=2)