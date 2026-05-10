PROJECT_DIR="/storage/student10/tungnl/Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis"

# Remove any existing broken symlinks in data/
rm -f "$PROJECT_DIR/data/MIMIC-CXR-JPG/files"
rm -f "$PROJECT_DIR/data/MIMIC-CXR/files"

# Create correct symlinks in data/
ln -s /storage/shared/MIMIC-CXR-JPG/split-4/files/ "$PROJECT_DIR/data/MIMIC-CXR-JPG/files"
ln -s /storage/shared/MIMIC-CXR-Report/download-mimic-cxr-txt/files/ "$PROJECT_DIR/data/MIMIC-CXR/files"

