import os
import pandas as pd
import numpy as np
import cv2
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib
#matplotlib.use('Agg')  # Disable GUI backend to avoid tkinter errors

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
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg','.png'))]
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
    labels = None
    if method == "dbscan":
        model = DBSCAN(eps=kwargs.get("eps", 0.5), min_samples=kwargs.get("min_samples", 5))
        labels = model.fit_predict(X)
    elif method == "optics":
        model = OPTICS(xi=kwargs.get("xi", 0.05), min_samples=kwargs.get("min_samples", 5), max_eps=kwargs.get("max_eps", np.inf))
        labels = model.fit_predict(X)
    elif method == "kmeans":
        model = KMeans(n_clusters=kwargs.get("n_clusters", 4), random_state=42, n_init=kwargs.get("n_init", 10)).fit(X)
        labels = model.labels_
    else:
        raise ValueError("Unsupported clustering method")

    
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
    plt.savefig(os.path.join(output_dir, 'tsne_projection.pdf'))
    plt.show()  
    plt.close()

def visualize_tsne_2d(X, labels, output_dir):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Apply t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    # Normalize labels to color map range
    unique_labels = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in labels])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=mapped_labels,
        s=40,
        alpha=0.8
    )
    plt.title("t-SNE Projection")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Add legend
    handles, _ = scatter.legend_elements()
    legend_labels = [f"Cluster {l}" if l != -1 else "Noise" for l in unique_labels]
    plt.legend(handles, legend_labels, title="Clusters", loc="best")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'tsne_2d.pdf'))
    plt.close()


def visualize_hsv_q2_3d(metadata, labels, output_dir):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import os

    # Extract Q2 (median) values for H, S, V
    X_q2 = np.array([
        [item["Hue Q2"], item["Sat Q2"], item["Val Q2"]]
        for item in metadata
    ])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X_q2[:, 0], X_q2[:, 1], X_q2[:, 2],
        c=labels,
        cmap='viridis',
        s=50,
        alpha=0.8
    )
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    ax.set_xlabel("Hue Q2")
    ax.set_ylabel("Saturation Q2")
    ax.set_zlabel("Value Q2")
    ax.set_title("HSV Q2 Quartiles (3D)")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hsv_q2_3d.pdf'))
    plt.show()  # 👈 Display the plot interactively
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
    plt.savefig(os.path.join(output_dir, 'clusters_visualization.pdf'))
    plt.close()

def create_cluster_collage(
    metadata,
    input_dir,
    output_path,
    images_per_cluster=16,
    thumb_size=(512, 512),
    grid_cols_per_cluster=8,
    clusters_per_row=4
):
    import math
    from collections import defaultdict

    cluster_dict = defaultdict(list)
    for item in metadata:
        cluster_dict[item["Cluster"]].append(item["Image Name"])

    collage_blocks = []

    sorted_cluster_ids = sorted([cid for cid in cluster_dict.keys() if cid != -1])
    for i, cluster_id in enumerate(sorted_cluster_ids + [-1]):  # Add noise at end
        images = cluster_dict[cluster_id]
        selected = images[:images_per_cluster]
        thumbnails = []

        for img_name in selected:
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, thumb_size)
            thumbnails.append(img)

        while len(thumbnails) < images_per_cluster:
            thumbnails.append(np.ones((thumb_size[1], thumb_size[0], 3), dtype=np.uint8) * 255)

        grid_rows = math.ceil(images_per_cluster / grid_cols_per_cluster)
        block_height = grid_rows * thumb_size[1]
        block_width = grid_cols_per_cluster * thumb_size[0]
        grid = np.ones((block_height, block_width, 3), dtype=np.uint8) * 255

        for idx, thumb in enumerate(thumbnails):
            r = idx // grid_cols_per_cluster
            c = idx % grid_cols_per_cluster
            grid[r * thumb_size[1]:(r + 1) * thumb_size[1], c * thumb_size[0]:(c + 1) * thumb_size[0]] = thumb

        label = "-1" if cluster_id == -1 else str(i + 1)  # 1-indexed clusters
        font_scale = 10
        thickness = 25
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = 10
        text_y = 10 + text_size[1]
        cv2.putText(
            grid,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        collage_blocks.append(grid)

    # Assemble final collage layout
    n_blocks = len(collage_blocks)
    layout_cols = clusters_per_row
    layout_rows = math.ceil(n_blocks / layout_cols)
    block_h, block_w = collage_blocks[0].shape[:2]
    full_collage = np.ones((layout_rows * block_h, layout_cols * block_w, 3), dtype=np.uint8) * 255

    for idx, block in enumerate(collage_blocks):
        r = idx // layout_cols
        c = idx % layout_cols
        full_collage[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w] = block

    cv2.imwrite(output_path, full_collage)
    print(f"Cluster collage saved to: {output_path}")


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
    input_dir = "/home/michal/Documents/Veronika Clustre/mix_dr0_idrid_ddr_messidor2"
    output_dir = "messidor-Clustering_multiclass"
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
        "eps": 0.59,
        "min_samples": 20,
        "n_clusters": 4,  # used only for kmeans
        "n_init": 20,
    }
    labels = run_clustering(X, method=clustering_method, **clustering_params)

    print("Visualizing clusters...")
    visualize_tsne(X, labels, output_dir)
    visualize_tsne_2d(X, labels, output_dir)
    visualize_hsv_q2_3d(metadata, labels, output_dir)
    visualize_clusters(X, labels, output_dir, title=f"{clustering_method.upper()} Clustering Results")

    print("Copying clustered images to folders...")
    metadata = copy_clustered_images(metadata, labels, input_dir, output_dir)
    print("Creating cluster collage...")
    create_cluster_collage(
        metadata,
        input_dir,
        output_path=os.path.join(output_dir, "cluster_collage.jpg"),
        grid_cols_per_cluster=6,
        clusters_per_row=2
    )



    print("Saving final metadata...")
    pd.DataFrame(metadata).to_csv(cache_file, index=False)

    summarize_clusters(labels, total_images=len(metadata))
    print(f"Done in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
