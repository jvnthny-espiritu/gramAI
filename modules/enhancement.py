import cv2

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_blur(image, method='gaussian', kernel_size=(5, 5)):
    if method == 'gaussian':
        return cv2.GaussianBlur(image, kernel_size, 0)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size[0])
    return image