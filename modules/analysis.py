import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return image

def compute_histogram(image):
    hist = {}
    for i, col in enumerate(['b', 'g', 'r']):
        hist[col] = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
    return hist

def compute_mean(image):
    return float(np.mean(image))

def compute_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    entropy = -np.sum([p * np.log2(p) for p in hist_norm if p > 0])
    return float(entropy)

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def analyze_directory(valid_files, before_dir, after_dir):
    results = []
    for idx, before_file in enumerate(valid_files):
        folder_name = os.path.basename(os.path.normpath(before_dir))
        after_file = f"{folder_name}_{idx+1:03d}.png"
        before_path = os.path.join(before_dir, before_file)
        after_path = os.path.join(after_dir, after_file)
        if not (os.path.exists(before_path) and os.path.exists(after_path)):
            continue
        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)
        if before_img is None or after_img is None:
            continue
        hist_before = compute_histogram(before_img)
        hist_after = compute_histogram(after_img)
        mean_before = compute_mean(before_img)
        mean_after = compute_mean(after_img)
        entropy_before = compute_entropy(before_img)
        entropy_after = compute_entropy(after_img)
        sharp_before = compute_sharpness(before_img)
        sharp_after = compute_sharpness(after_img)
        results.append({
            "before_file": before_file,
            "after_file": after_file,
            "mean_before": mean_before,
            "mean_after": mean_after,
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "sharpness_before": sharp_before,
            "sharpness_after": sharp_after,
            "hist_before": hist_before,
            "hist_after": hist_after,
            "before_img": before_img,
            "after_img": after_img
        })
    return results

def visualize_analysis(results, sample_indices=None):
    if not results:
        print("No results to visualize.")
        return

    if sample_indices is None:
        sample_indices = list(range(min(5, len(results))))

    for idx in sample_indices:
        res = results[idx]
        before_img = cv2.cvtColor(res["before_img"], cv2.COLOR_BGR2RGB)
        after_img = cv2.cvtColor(res["after_img"], cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Analysis for: {res['before_file']} -> {res['after_file']}", fontsize=16)

        # Show images
        axes[0, 0].imshow(before_img)
        axes[0, 0].set_title("Before")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(after_img)
        axes[0, 1].set_title("After")
        axes[0, 1].axis('off')

        # Show metrics
        metrics_text = (
            f"Mean: {res['mean_before']:.2f} → {res['mean_after']:.2f}\n"
            f"Entropy: {res['entropy_before']:.2f} → {res['entropy_after']:.2f}\n"
            f"Sharpness: {res['sharpness_before']:.2f} → {res['sharpness_after']:.2f}"
        )
        axes[0, 2].text(0.1, 0.5, metrics_text, fontsize=12, va='center')
        axes[0, 2].axis('off')

        # Show histograms
        for i, col in enumerate(['b', 'g', 'r']):
            axes[1, 0].plot(res['hist_before'][col], color=col)
            axes[1, 1].plot(res['hist_after'][col], color=col)
        axes[1, 0].set_title("Histogram Before")
        axes[1, 1].set_title("Histogram After")
        axes[1, 2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()