import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import shutil
from tqdm import tqdm


from collections import defaultdict
import plotly.graph_objects as go

def plot_hue_values(median_hues):
    value_counts = defaultdict(int)
    x = []
    y = []

    for val in median_hues:
        x.append(val)
        y.append(value_counts[val])
        value_counts[val] += 1

    fig = go.Figure(data=go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=10, color='orange')
    ))

    fig.update_layout(
        title='Median Hue Values per Image (Duplicates Stacked)',
        xaxis_title='Median Hue',
        yaxis_title='Occurrence Index'
    )
    fig.show()

# Parameters
INPUT_DIR = '/media/michal/Krtko a Noz/DDR-dataset/lesion_segmentation/train/image_crp'  # Change this to your dataset path
OUTPUT_DIR = 'clustered_images'
HUE_EPS = 0.2914  # DBSCAN epsilon (tunable)
MIN_SAMPLES = 77  # Minimum points in a cluster

def load_images(image_dir):
    image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    return image_paths

def get_median_hue(image_path, mask_dir):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path.name}")
        return None
    
    # Use same filename for mask, different directory
    mask_path = Path(mask_dir) / image_path.name
    if not mask_path.exists():
        print(f"Mask not found for {image_path.name}")
        return None
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != img.shape[:2]:
        print(f"Invalid mask for {image_path.name}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]

    # Mask application: keep hue values only where mask is white (>0)
    fundus_hue = hue[mask > 0]
    if fundus_hue.size == 0:
        print(f"No fundus pixels found in {image_path.name}")
        return None
    
    median_hue = np.mean(fundus_hue)
    return median_hue

def cluster_hues(hue_values):
    X = np.array(hue_values).reshape(-1, 1)
    clustering = DBSCAN(eps=HUE_EPS, min_samples=MIN_SAMPLES).fit(X)
    return clustering.labels_

def save_clustered_images(image_paths, labels):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        cluster_dir = os.path.join(OUTPUT_DIR, f'cluster_{label}')
        os.makedirs(cluster_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(cluster_dir, img_path.name))

def main():
    MASK_DIR = '/media/michal/Krtko a Noz/DDR-dataset/lesion_segmentation/train/image_crp'

    print("Loading images...")
    image_paths = load_images(INPUT_DIR)
    
    print("Calculating median hue values...")
    median_hues = []
    valid_paths = []

    for path in tqdm(image_paths):
        median_hue = get_median_hue(path, MASK_DIR)
        if median_hue is not None:
            median_hues.append(median_hue)
            valid_paths.append(path)
    
    print(f"Clustering {len(median_hues)} images using DBSCAN...")
    labels = cluster_hues(median_hues)
    
    print("Saving clustered images to folders...")
    save_clustered_images(valid_paths, labels)
    
    print("Done!")
    print("Plotting hue values...")
    plot_hue_values(median_hues)
if __name__ == "__main__":
    main()
