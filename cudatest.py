import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Temporary workaround
# Your imports here

import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current CUDA Device:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
