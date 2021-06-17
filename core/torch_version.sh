# import torch; version, cu = torch.__version__.split('+'); version = version[:-1] + '0'; TORCH_GEOM_PATH = f'https://pytorch-geometric.com/whl/torch-{version}+{cu}.html'; print(TORCH_GEOM_PATH)"
# torch==1.8.0, cuda==10.2 in the server
pip3 install torch==1.8.0
pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip3 install torch-geometric