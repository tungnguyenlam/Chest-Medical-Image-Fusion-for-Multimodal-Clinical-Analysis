import os
import sys
import kagglehub

# 1. Check if the command is being run in the project root, exit if not.
if not os.path.exists('.git') or not os.path.exists('scripts/dataset-download/kaggle-mimic-cxr-jpg-setup.py'):
    print("Error: This script must be run from the project root directory.")
    sys.exit(1)

# 2. Download the dataset
print("Downloading simhadrisadaram/mimic-cxr-dataset via kagglehub...")
path = kagglehub.dataset_download("simhadrisadaram/mimic-cxr-dataset")
print(f"Path to dataset files: {path}")

# 3. Symlink back to data/data-kaggle
target_link = "data/data-kaggle"

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# If target_link exists as an empty directory, remove it to make way for the symlink
if os.path.exists(target_link) and os.path.isdir(target_link) and not os.path.islink(target_link):
    if not os.listdir(target_link):
        os.rmdir(target_link)
    else:
        print(f"Warning: {target_link} is not empty. Symlinking inside it.")
        target_link = os.path.join(target_link, os.path.basename(path))

# Remove existing symlink or file if it exists at the target_link path
if os.path.lexists(target_link):
    os.remove(target_link)

print(f"Creating symlink: {target_link} -> {path}")
os.symlink(path, target_link)
print("Setup complete.")