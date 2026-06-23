tmux

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
source ~/miniforge3/bin/activate
echo 'source ~/miniforge3/bin/activate' >> ~/.bashrc

conda create -n camchex python=3.13 libjpeg-turbo -y
conda activate camchex
conda install -c conda-forge p7zip -y       # only needed to (un)bundle subsets

pip install uv

git clone https://github.com/tungnguyenlam/Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis.git
cd Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
uv pip install -r requirements.txt

curl -fsSL https://herdr.dev/install.sh | sh
export PATH="/root/.local/bin:$PATH"

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

git submodule update --init --recursive mimic-cxr
