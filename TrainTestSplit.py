import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = r"C:\Users\bilal\Downloads\Train-224x224\train"
csv_path = r"C:\Users\bilal\Downloads\diabetic-retinopathy-detection\trainLabels.csv\trainLabels.csv"
test_dir = r"C:\Users\bilal\Downloads\Test_Set-224x224"

# Split ratio
test_ratio = 0.15  # 15% test

# Create test directory
os.makedirs(test_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Ensure correct column names
df.columns = ["filename", "class"]

# Strip spaces
df["filename"] = df["filename"].astype(str).str.strip()

# Convert class column to integer
df["class"] = df["class"].astype(int)

# Stratified split
train_df, test_df = train_test_split(
    df,
    test_size=test_ratio,
    stratify=df["class"],
    random_state=42
)

# Supported extensions
extensions = [".jpeg", ".jpg", ".png"]

# Create a set of actual filenames in dataset
existing_files = set(os.listdir(dataset_dir))

# Function to find actual file for a given base name
def find_file(base_name):
    for ext in extensions:
        candidate = base_name + ext
        if candidate in existing_files:
            return candidate
    return None

# Move test images
missing_files = 0

for _, row in test_df.iterrows():
    base_name = row["filename"]
    actual_file = find_file(base_name)
    if actual_file:
        src = os.path.join(dataset_dir, actual_file)
        dst = os.path.join(test_dir, actual_file)
        shutil.move(src, dst)
    else:
        print(f"Warning: {base_name} not found with any supported extension")
        missing_files += 1

# Save updated CSVs
train_df.to_csv("train_labels.csv", index=False)
test_df.to_csv("test_labels.csv", index=False)

print(f"Total test images moved: {len(test_df) - missing_files}")
print(f"Missing files: {missing_files}")