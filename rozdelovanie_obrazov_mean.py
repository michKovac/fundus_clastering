import os
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, KMeans
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def process_image(file_info):
    """Process a single image to extract hue"""
    file_path, file_name = file_info

    image = cv2.imread(file_path)
    if image is None:
        return None
    
    # For fundus images, we need to get more differentiation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_image = hsv_image[:,:,0]
    s_image = hsv_image[:,:,1]
    v_image = hsv_image[:,:,2]
    avg_hue = np.mean(h_image)
    avg_sat = np.mean(s_image)
    avg_val = np.mean(v_image)
    median_hue = np.median(h_image)
    median_sat = np.median(s_image)
    median_val = np.median(v_image)
    hue_q = np.array([np.percentile(h_image, 25), np.percentile(h_image, 50), np.percentile(h_image, 75)])
    s_q = np.array([np.percentile(s_image, 25), np.percentile(s_image, 50), np.percentile(s_image, 75)])
    v_q = np.array([np.percentile(v_image, 25), np.percentile(v_image, 50), np.percentile(v_image, 75)])

    return {
        "Image Name": file_name, 
        "Average Hue": float(avg_hue),
        "Average Saturation": float(avg_sat),
        "Average Value": float(avg_val),
        "Median Hue": float(median_hue),
        "Median Saturation": float(median_sat),
        "Median Value": float(median_val),
        "Hue Quartiles": hue_q,
        "Saturation Quartiles": s_q,
        "Value Quartiles": v_q
    }


def main():
    print(f"Starting DBSCAN clustering process for fundus images")
    print(f"Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-03-28 17:11:27")
    print(f"Current User's Login: AvetyNS")
    start_time = time.time()
    
    # Define directories
    input_dir = "/home/michal/Downloads/DR_grading/sorted_train/0"
    output_base_dir = "DDR-DBSCAN"
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Define CSV path for caching HSV data
    hsv_cache_file = os.path.join(output_base_dir, "image_metadata.csv")
    
    # Check if HSV cache file exists

    print("No cached HSV data found, processing images...")
    # Get all JPG files
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    total_files = len(jpg_files)
    print(f"Found {total_files} images to process")
    
    # Prepare file path + name tuples
    file_tuples = [(os.path.join(input_dir, f), f) for f in jpg_files]
    
    # Extract HSV values using parallel processing
    print("Extracting HSV values from images...")
    metadata_list = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, result in enumerate(executor.map(process_image, file_tuples)):
            if i % max(1, total_files//20) == 0:
                print(f"Progress: {i/total_files*100:.1f}% completed")
            if result:
                metadata_list.append(result)
    
    print(f"Successfully processed {len(metadata_list)}/{total_files} images")
    
    # Save HSV data for future use
    print(f"Saving HSV data to {hsv_cache_file}...")
    pd.DataFrame(metadata_list).to_csv(hsv_cache_file, index=False)
    
    valid_images = len(metadata_list)
    
    # For fundus images, we need to use very small eps and include more features
    hue_values = np.array([item["Average Hue"] for item in metadata_list])
    sat_values = np.array([item["Average Saturation"] for item in metadata_list])
    val_values = np.array([item["Average Value"] for item in metadata_list])
    median_hues = np.array([item["Median Hue"] for item in metadata_list])
    median_sats = np.array([item["Median Saturation"] for item in metadata_list])
    median_vals = np.array([item["Median Value"] for item in metadata_list])
    hue_quartiles = np.array([item["Hue Quartiles"] for item in metadata_list])
    s_quartiles = np.array([item["Saturation Quartiles"] for item in metadata_list])
    v_quartiles = np.array([item["Value Quartiles"] for item in metadata_list])
    
    # Convert hue to circular coordinates
    hue_radians = np.deg2rad(median_hues * 2)
    
    # New feature matrix with quartiles for hue
    X = np.column_stack([
        hue_quartiles[:,0],  # Q1 Hue
        hue_quartiles[:,1],  # Q2 Hue
        hue_quartiles[:,2],  # Q3 Hue
        s_quartiles[:,0],    # Q1 Saturation
        s_quartiles[:,1],    # Q2 Saturation
        s_quartiles[:,2],    # Q3 Saturation
        v_quartiles[:,0],    # Q1 Value
        v_quartiles[:,1],    # Q2 Value
        v_quartiles[:,2],    # Q3 Value
    ])
    # X = np.column_stack([
    #     np.cos(hue_radians) * 2.0,  # Higher weight for hue to improve separation
    #     np.sin(hue_radians) * 2.0,  # Higher weight for hue to improve separation
    #     median_sats / 255,     # Higher weight for saturation
    #     median_vals / 255      # Higher weight for value
    # ])

    # neigh = NearestNeighbors(n_neighbors=5)
    # nbrs = neigh.fit(X)
    # distances, _ = nbrs.kneighbors(X)
    
    # k_distances = np.sort(distances[:, -1])
    # plt.plot(k_distances)
    # plt.xlabel("Bod")
    # plt.ylabel("Vzdialenosť k 5. susedovi")
    # plt.title("k-distance graf pre DBSCAN (k=5)")
    # plt.grid(True)
    # plt.show()



    # Apply standard scaling
    X = StandardScaler().fit_transform(X)
    
    # Analyze the data distribution before clustering
    print("Analyzing data distribution...")
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(hue_values, bins=30, color='skyblue', edgecolor='black')
    plt.title('Hue Distribution')
    plt.xlabel('Hue (0-180)')
    
    plt.subplot(2, 2, 2)
    plt.hist(sat_values, bins=30, color='lightgreen', edgecolor='black')
    plt.title('Saturation Distribution')
    plt.xlabel('Saturation (0-255)')
    
    plt.subplot(2, 2, 3)
    plt.hist(val_values, bins=30, color='salmon', edgecolor='black')
    plt.title('Value Distribution')
    plt.xlabel('Value (0-255)')
    
    plt.subplot(2, 2, 4)
    plt.scatter(X[:, 0], X[:, 1], c=hue_values, cmap='hsv', s=30, alpha=0.6)
    plt.title('Hue on Unit Circle')
    plt.xlabel('Cosine Component')
    plt.ylabel('Sine Component')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, 'data_distribution.png'))
    plt.close()
    
    # DBSCAN parameters
    eps = 0.6
    min_samples = 50
    k = 4
    print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    #dbscan = OPTICS(xi=0.7, max_eps=0.85)
    #dbscan = HDBSCAN()
    labels = dbscan.fit_predict(X)
    #kmeans = KMeans(n_clusters=k, n_init=20, random_state=81).fit(X)
    #labels = kmeans.labels_
    from sklearn.manifold import TSNE
    
    # t-SNE to reduce to 2D
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    X_3d = tsne.fit_transform(X)
    
    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels, cmap='viridis', s=50)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    
    ax.set_title("t-SNE Projection from 9D to 3D")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.show()
    
    # Count clusters
    unique_labels = np.unique(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = np.sum(labels == -1)
    
    print(f"DBSCAN results: {n_clusters} clusters and {n_noise} noise points ({n_noise/valid_images*100:.1f}%)")
    
    # If we still don't have multiple clusters, show error
    if n_clusters <= 1:
        print(f"Value ERROR - Only {n_clusters} cluster formed. Try different parameters.")
    
    # Print cluster sizes
    for label in sorted(unique_labels):
        count = np.sum(labels == label)
        name = f"Cluster_{label}" if label >= 0 else "Noise"
        print(f"{name}: {count} images ({count/valid_images*100:.1f}%)")
    
    # Visualize the results
    plt.figure(figsize=(10, 8))
    
    # Plot points colored by cluster
    for label in unique_labels:
        if label == -1:
            # Black used for noise
            color = [0, 0, 0, 1]
            marker = 'x'
        else:
            # Generate a color based on the cluster label
            color = plt.cm.rainbow(float(label) / max(1, max(unique_labels)))
            marker = 'o'
        
        mask = labels == label
        plt.scatter(
            X[mask, 0], X[mask, 1],
            c=[color], marker=marker, s=30, alpha=0.7,
            label=f'Cluster {label}' if label != -1 else 'Noise'
        )
    
    plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
    plt.xlabel('Cosine Component')
    plt.ylabel('Sine Component')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_base_dir, 'clusters_visualization.png'))
    plt.close()
    
    # Add cluster info to metadata
    for i, item in enumerate(metadata_list):
        item["Cluster"] = int(labels[i])
    
    # Create cluster directories
    cluster_dirs = {}
    for label in unique_labels:
        name = f"Cluster_{label}" if label >= 0 else "Noise"
        cluster_dir = os.path.join(output_base_dir, name)
        os.makedirs(cluster_dir, exist_ok=True)
        cluster_dirs[label] = cluster_dir
    
    # Copy files to appropriate directories
    print("Copying files to cluster directories...")
    
    def copy_file(item):
        try:
            cluster = item["Cluster"]
            src = os.path.join(input_dir, item["Image Name"])
            dst = os.path.join(cluster_dirs[cluster], item["Image Name"])
            shutil.copy2(src, dst)
            return True
        except:
            return False
    
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count()*2)) as executor:
        copied = sum(executor.map(copy_file, metadata_list))
    
    # Save updated metadata with cluster info
    pd.DataFrame(metadata_list).to_csv(hsv_cache_file, index=False)
    
    # Report completion
    elapsed_time = time.time() - start_time
    print(f"Processing complete in {elapsed_time:.1f} seconds")
    print(f"Created {n_clusters} clusters using DBSCAN with eps={eps}")

if __name__ == "__main__":
    main()