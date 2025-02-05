#!/usr/bin/env python3

# Print diagnostic information to determine whether Torch and CUDA are
# properly installed.

import torch

print("torch.cuda.is_available() : {}".format(torch.cuda.is_available()))

n = torch.cuda.device_count()

for i in range(0, n):
    print(
        "torch.cuda.get_device_name({}) : {}".format(i, torch.cuda.get_device_name(0))
    )

print("torch.cuda.current_device() : {}".format(torch.cuda.current_device()))
