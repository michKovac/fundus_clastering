import os
import pandas as pd
import numpy as np
import cv2
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def process_image(file_info):
    file_path, file_name = file_info
    image = cv2.imread(file_path)
    if image is None:
        return None

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]

    return {
        "Image Name": file_name,
        "Average Hue": float(np.mean(h)),
        "Average Saturation": float(np.mean(s)),
        "Average Value": float(np.mean(v)),
        "Median Hue": float(np.median(h)),
        "Median Saturation": float(np.median(s)),
        "Median Value": float(np.median(v)),
        "Hue Q1": float(np.percentile(h, 25)),
        "Hue Q2": float(np.percentile(h, 50)),
        "Hue Q3": float(np.percentile(h, 75)),
        "Sat Q1": float(np.percentile(s, 25)),
        "Sat Q2": float(np.percentile(s, 50)),
        "Sat Q3": float(np.percentile(s, 75)),
        "Val Q1": float(np.percentile(v, 25)),
        "Val Q2": float(np.percentile(v, 50)),
        "Val Q3": float(np.percentile(v, 75)),
    }


def load_or_process_images(input_dir, cache_file):
    if os.path.exists(cache_file):
        print("Loading HSV data from cache...")
        return pd.read_csv(cache_file).to_dict(orient="records")

    print("No cached HSV data found, processing images...")
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    file_tuples = [(os.path.join(input_dir, f), f) for f in jpg_files]

    metadata_list = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = executor.map(process_image, file_tuples)
        for result in tqdm(futures, total=len(file_tuples), desc="Processing images"):
            if result:
                metadata_list.append(result)

    pd.DataFrame(metadata_list).to_csv(cache_file, index=False)
    return metadata_list


def extract_features(metadata_list, mode="quartiles"):
    if mode == "quartiles":
        X = np.array([
            [
                item["Hue Q1"], item["Hue Q2"], item["Hue Q3"],
                item["Sat Q1"], item["Sat Q2"], item["Sat Q3"],
                item["Val Q1"], item["Val Q2"], item["Val Q3"]
            ]
            for item in metadata_list
        ])
    elif mode == "hsv_median":
        X = np.array([
            [
                item["Median Hue"], item["Median Saturation"], item["Median Value"]
            ]
            for item in metadata_list
        ])
    elif mode == "hue_circle":
        X = np.array([
            [
                np.cos(np.deg2rad(item["Median Hue"] * 2)),  # Hue on circle
                np.sin(np.deg2rad(item["Median Hue"] * 2)),
                item["Median Saturation"] / 255,
                item["Median Value"] / 255
            ]
            for item in metadata_list
        ])
    elif mode == "hue_quartiles":
        X = np.array([
            [item["Hue Q1"], item["Hue Q2"], item["Hue Q3"]]
            for item in metadata_list
        ])
    else:
        raise ValueError(f"Unsupported feature mode: {mode}")

    return StandardScaler().fit_transform(X)



def run_clustering(X, method="dbscan", **kwargs):
    if method == "dbscan":
        model = DBSCAN(eps=kwargs.get("eps", 0.5), min_samples=kwargs.get("min_samples", 5))
    elif method == "optics":
        model = OPTICS(xi=kwargs.get("xi", 0.05), min_samples=kwargs.get("min_samples", 5), max_eps=kwargs.get("max_eps", np.inf))
    elif method == "kmeans":
        model = KMeans(n_clusters=kwargs.get("n_clusters", 4), random_state=42, n_init=10)
    else:
        raise ValueError("Unsupported clustering method")

    labels = model.fit_predict(X)
    return labels


def visualize_tsne(X, labels, output_dir):
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    X_3d = tsne.fit_transform(X)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels, cmap='viridis', s=50)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    plt.title("t-SNE Projection (3D)")
    plt.savefig(os.path.join(output_dir, 'tsne_projection.png'))
    plt.close()


def visualize_clusters(X, labels, output_dir, title):
    unique_labels = np.unique(labels)
    plt.figure(figsize=(10, 8))

    for label in unique_labels:
        color = [0, 0, 0, 1] if label == -1 else plt.cm.rainbow(float(label) / max(1, max(unique_labels)))
        marker = 'x' if label == -1 else 'o'
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], marker=marker, s=30, alpha=0.7, label=f'Cluster {label}' if label != -1 else 'Noise')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'clusters_visualization.png'))
    plt.close()


def copy_clustered_images(metadata_list, labels, input_dir, output_dir):
    total = len(metadata_list)

    # Add labels to metadata
    for i, item in enumerate(metadata_list):
        item["Cluster"] = int(labels[i])

    # Count images per cluster
    from collections import Counter
    cluster_counts = Counter(labels)

    # Create folders with descriptive names
    cluster_dirs = {}
    for label, count in cluster_counts.items():
        percentage = (count / total) * 100
        folder_name = f"Cluster {label}: {count} images ({percentage:.1f}%)" if label >= 0 else f"Noise: {count} images ({percentage:.1f}%)"
        cluster_path = os.path.join(output_dir, folder_name)
        os.makedirs(cluster_path, exist_ok=True)
        cluster_dirs[label] = cluster_path

    def copy_file(item):
        try:
            src = os.path.join(input_dir, item["Image Name"])
            dst = os.path.join(cluster_dirs[item["Cluster"]], item["Image Name"])
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Failed to copy {item['Image Name']}: {e}")

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count()*2)) as executor:
        executor.map(copy_file, metadata_list)

    return metadata_list


def summarize_clusters(labels, total_images):
    unique_labels = np.unique(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = np.sum(labels == -1)

    print(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points ({n_noise / total_images * 100:.1f}%)")
    for label in sorted(unique_labels):
        count = np.sum(labels == label)
        name = f"Cluster {label}" if label >= 0 else "Noise"
        print(f"{name}: {count} images ({count / total_images * 100:.1f}%)")


def main():
    input_dir = "/home/michal/Downloads/DR_grading/sorted_train/0"
    output_dir = "DDR-Clustering"
    cache_file = os.path.join(output_dir, "image_metadata.csv")

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    print("Loading and processing images...")
    metadata = load_or_process_images(input_dir, cache_file)

    print("Extracting features...")
    X = extract_features(metadata,  mode="quartiles")

    print("Clustering images...")
    # CHANGE THIS LINE TO SWITCH METHODS
    clustering_method = "dbscan"  # options: 'dbscan', 'optics', 'kmeans'
    clustering_params = {
        "eps": 0.6,
        "min_samples": 50,
        "n_clusters": 4,  # used only for kmeans
    }
    labels = run_clustering(X, method=clustering_method, **clustering_params)

    print("Visualizing clusters...")
    visualize_tsne(X, labels, output_dir)
    visualize_clusters(X, labels, output_dir, title=f"{clustering_method.upper()} Clustering Results")

    print("Copying clustered images to folders...")
    metadata = copy_clustered_images(metadata, labels, input_dir, output_dir)

    print("Saving final metadata...")
    pd.DataFrame(metadata).to_csv(cache_file, index=False)

    summarize_clusters(labels, total_images=len(metadata))
    print(f"Done in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
