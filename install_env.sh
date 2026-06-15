pip install "setuptools<82" wheel

conda install -y \
  pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 \
  "mkl<2025" "intel-openmp<2025" \
  -c pytorch -c nvidia
conda install -y -c conda-forge "cmake<4" ninja ffmpeg

pip install diffusers==0.35.2 transformers==4.43.0 tokenizers==0.19.1 opencv-python==4.13.0.92 \
  accelerate==1.12.0 safetensors==0.7.0 peft==0.18.1

pip install ftfy tensorboard Jinja2 sentencepiece cairosvg segment_anything prodigyopt scikit-image wandb gdown

git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive

pip install svgwrite svgpathtools cssutils numba
pip install --no-build-isolation visdom
pip install torch-tools

python setup.py install

cd ..
