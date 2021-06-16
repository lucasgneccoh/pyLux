import torch
version, cu = torch.__version__.split('+')
version = version[:-1] + '0'
TORCH_GEOM_PATH = f"https://pytorch-geometric.com/whl/torch-{version}+{cu}.html"
print(TORCH_GEOM_PATH)