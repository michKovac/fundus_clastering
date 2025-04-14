# Fundus Image Clustering by Color Similarity

This project performs **unsupervised clustering** of fundus images based on their **color characteristics** (in HSV color space). It extracts statistical features from the images, applies clustering algorithms (DBSCAN, OPTICS, KMeans), and organizes similar images into folders based on their color similarity.

---

## 🧠 Project Overview

Fundus images can vary significantly in color due to lighting, camera settings, and patient physiology. Grouping these images by color distribution helps in:

- Preprocessing for machine learning tasks
- Quality control and outlier detection
- Dataset curation and visualization

This tool allows you to:
- Extract HSV-based features from images
- Apply clustering algorithms on those features
- Visualize the results with t-SNE and 2D projections
- Organize images into cluster folders

---

## 🗂️ Project Structure

```
project/
│
├── DDR-Clustering/               # Output directory
│   ├── Cluster X: N images .../  # Clustered image folders
│   ├── tsne_projection.png       # 3D t-SNE visualization
│   └── clusters_visualization.png
│
├── image_metadata.csv            # Cached HSV feature data
└── cluster_fundus_images.py      # Main script (shown above)
```

---

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install numpy pandas opencv-python scikit-learn matplotlib tqdm
```

### 2. Run the Clustering Script

Modify the `input_dir` in the `main()` function to point to your image folder, then run:

```bash
python cluster_fundus_images.py
```

### 3. Output

- Clusters saved in separate folders
- `image_metadata.csv`: feature stats
- Cluster visualizations: `tsne_projection.png`, `clusters_visualization.png`

---

## ⚙️ Configuration

You can modify the following parameters in `main()`:

```python
clustering_method = "dbscan"  # Options: 'dbscan', 'optics', 'kmeans'
clustering_params = {
    "eps": 0.6,
    "min_samples": 50,
    "n_clusters": 4,  # Only used for kmeans
}
feature_mode = "quartiles"  # Options: 'quartiles', 'hsv_median', 'hue_circle', 'hue_quartiles'
```

---

## 📊 Feature Modes

| Mode          | Description                                      |
|---------------|--------------------------------------------------|
| `quartiles`   | HSV Q1, Q2, Q3 values                            |
| `hsv_median`  | Median HSV values                                |
| `hue_circle`  | Hue mapped to circular space + normalized S/V    |
| `hue_quartiles` | Only hue quartiles                             |

---

## 📈 Clustering Methods

- **DBSCAN**: Density-based, great for finding arbitrary shaped clusters
- **OPTICS**: Similar to DBSCAN, handles varying density
- **KMeans**: Requires setting number of clusters, assumes spherical clusters

---

## 📷 Example Results

![t-SNE Projection](DDR-Clustering/tsne_projection.png)
*3D t-SNE of clustered fundus images*

---

## 🧪 Tips

- Adjust `eps` and `min_samples` for DBSCAN/OPTICS depending on dataset size.
- Use `kmeans` when you know approximately how many color groups you expect.
- Use `image_metadata.csv` to analyze image statistics before/after clustering.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

