# Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis

```
# Install (Go binary, runs on macOS)
# Download from: https://github.com/ygidtu/NBIA_data_retriever_CLI

# Run with multiple parallel processes
./NBIA_data_retriever_CLI -i your_manifest.tcia -s ./output -p 4
```

# Quick setup

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash Miniforge3-$(uname)-$(uname -m).sh -b

source miniforge3/bin/activate

conda create -n tung python=3.13 -y

pip install uv

git clone https://github.com/tungnguyenlam/Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis.git

uv pip install requirements.txt

```