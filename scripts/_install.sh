python -m pip install -f https://download.pytorch.org/whl/cu113/torch_stable.html \
    torch==1.10.0+cu113 \
    torchvision==0.11.0+cu113 \
    torchaudio==0.10.0+cu113 \
    torchtext==0.11.0

python -m pip install -r requirements.txt
python -m src.verify_installation
