import torch
from torchvision import transforms


inversed_transform = transforms.Compose([
    # transforms.Resize((299, 299)),
    transforms.Normalize((-1., -1., -1.), (2., 2., 2.)),
    transforms.ConvertImageDtype(torch.uint8)
])

inversed_transform_ir = transforms.Compose([
    transforms.Normalize((-1.,), (2.,)),
    transforms.ConvertImageDtype(torch.uint8)
])


class InverseNormalize(transforms.Normalize):
    def __init__(self, mean=[-1., -1., -1.], std=[2., 2., 2.]):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())