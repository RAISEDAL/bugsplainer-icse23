import sys

import torch

assert sys.version_info[:2] == (3, 9)
assert torch.cuda.is_available()
assert torch.version.cuda == '11.3'
