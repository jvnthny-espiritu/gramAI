import cv2
import numpy as np
import random
import os

def rotate_90_180_270(image):
    """Return a list of images rotated by 90, 180, and 270 degrees."""
    return [
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_180),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

def flip_all(image):
    """Return a list of all flips: horizontal, vertical, both."""
    return [
        cv2.flip(image, 0),   # vertical
        cv2.flip(image, 1),   # horizontal
        cv2.flip(image, -1)   # both
    ]

def random_scale(image, scale_range=(0.8, 1.2)):
    scale = random.uniform(*scale_range)
    h, w = image.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    scaled = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # Center crop or pad to original size
    if scale < 1.0:
        pad_h = (h - nh) // 2
        pad_w = (w - nw) // 2
        out = np.zeros_like(image)
        out[pad_h:pad_h+nh, pad_w:pad_w+nw] = scaled
        return out
    else:
        start_h = (nh - h) // 2
        start_w = (nw - w) // 2
        return scaled[start_h:start_h+h, start_w:start_w+w]

def random_color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2):
    img = image.astype(np.float32) / 255.0
    # Brightness
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        img = np.clip(img * factor, 0, 1)
    # Contrast
    if contrast > 0:
        mean = img.mean(axis=(0,1), keepdims=True)
        factor = 1.0 + random.uniform(-contrast, contrast)
        img = np.clip((img - mean) * factor + mean, 0, 1)
    # Saturation
    if saturation > 0:
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[...,1] = np.clip(hsv[...,1] * (1.0 + random.uniform(-saturation, saturation)), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return (img * 255).astype(np.uint8)

def random_gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def random_cutout(image, max_h_frac=0.3, max_w_frac=0.3):
    h, w = image.shape[:2]
    cutout_h = int(h * random.uniform(0, max_h_frac))
    cutout_w = int(w * random.uniform(0, max_w_frac))
    y = random.randint(0, h - cutout_h)
    x = random.randint(0, w - cutout_w)
    image = image.copy()
    image[y:y+cutout_h, x:x+cutout_w] = np.random.randint(0, 256, (cutout_h, cutout_w, 3), dtype=np.uint8)
    return image

def augment_image(image):
    # Only randomize scale, color jitter, noise, cutout
    aug_funcs = [
        lambda img: random_scale(img, scale_range=(0.8, 1.2)),
        lambda img: random_color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2),
        lambda img: random_gaussian_noise(img, std=10),
        random_cutout
    ]
    random.shuffle(aug_funcs)
    aug_img = image.copy()
    for func in aug_funcs:
        if random.random() < 0.7:  # 70% chance to apply each augmentation
            aug_img = func(aug_img)
    return aug_img

def augment_and_save(image, save_dir, base_name, n_augments=5):
    os.makedirs(save_dir, exist_ok=True)
    # Save fixed rotations
    for angle, rot_img in zip([90, 180, 270], rotate_90_180_270(image)):
        save_path = os.path.join(save_dir, f"{base_name}_rot_{angle}.png")
        cv2.imwrite(save_path, rot_img)
    # Save fixed flips
    for flip_name, flip_img in zip(['v', 'h', 'hv'], flip_all(image)):
        save_path = os.path.join(save_dir, f"{base_name}_flip_{flip_name}.png")
        cv2.imwrite(save_path, flip_img)
    # Save random augmentations
    for i in range(n_augments):
        aug_img = augment_image(image)
        save_path = os.path.join(save_dir, f"{base_name}_aug_{i+1}.png")
        cv2.imwrite(save_path, aug_img)