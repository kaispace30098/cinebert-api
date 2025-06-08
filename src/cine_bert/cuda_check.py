import torch

# 1. Is CUDA available at all?
print("CUDA available:", torch.cuda.is_available())

# 2. How many CUDA devices can PyTorch see?
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    # 3. Which device is the current default?
    current_idx = torch.cuda.current_device()
    print("Current device index:", current_idx)

    # 4. What’s its name?
    print("Current device name:", torch.cuda.get_device_name(current_idx))

    # 5. Is cuDNN enabled?
    print("cuDNN enabled:", torch.backends.cudnn.enabled)
