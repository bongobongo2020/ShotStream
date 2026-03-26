git clone https://github.com/KlingAIResearch/ShotStream.git
cd ShotStream
conda create -n shotstream python=3.10 -y
conda activate shotstream
conda install nvidia/label/cuda-12.4.1::cuda
conda install -c nvidia/label/cuda-12.4.1 cudatoolkit
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation