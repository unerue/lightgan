import torch
from torchvision import transforms


inversed_transform = transforms.Compose([
    # transforms.Resize((299, 299)),
    transforms.Normalize((-1., -1., -1.), (2., 2., 2.)),
    transforms.ConvertImageDtype(torch.uint8)
])