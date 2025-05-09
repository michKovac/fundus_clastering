import os
import shutil
import pandas as pd

# === Configuration ===
csv_path = '/home/michal/Documents/Veronika Clustre/messidor_data.csv'  # path to your CSV file
images_folder = '/home/michal/Documents/Veronika Clustre/messidor-2/preprocess'  # path to folder with all images
output_folder = '/home/michal/Documents/Veronika Clustre/messidorsorted'  # where to store the sorted images

# === Step 1: Load and filter the CSV ===
df = pd.read_csv(csv_path)
gradable_df = df[df['adjudicated_gradable'] == 1]

# === Step 2: Sort images into folders ===
for _, row in gradable_df.iterrows():
    image_name = row['id_code']
    diagnosis = row['diagnosis']

    # Create target folder path
    target_folder = os.path.join(output_folder, str(diagnosis))
    os.makedirs(target_folder, exist_ok=True)

    # Full paths for source and destination
    source_path = os.path.join(images_folder, image_name)
    target_path = os.path.join(target_folder, image_name)

    # Move/copy the image if it exists
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)
    else:
        print(f"Image not found: {source_path}")

print("Image sorting complete.")