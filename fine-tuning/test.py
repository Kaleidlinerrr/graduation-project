import torch
print(f"Pytorch version:{torch.__version__}")
print(f"CUDA availabel:{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(torch.version.cuda)
    print(torch.cuda.device_count())