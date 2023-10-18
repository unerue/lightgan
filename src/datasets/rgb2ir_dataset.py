import os.path
import random
import glob
from PIL import Image
from typing import Optional, Any

from torchvision import transforms
from torch.utils.data import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


def get_transform(
    resize: int = 286, crop_size: int = 256, grayscale: bool = False, train: bool = True
):
    """
    preprocessing = resize_and_crop
    load_size = 286, crop_size=256
    return -1 to 1
    """
    transform_list = []
    if train:
        transform_list.append(transforms.Resize(resize))
        transform_list.append(transforms.RandomCrop(crop_size))
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor()]
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


class UnalignedRgb2IrDataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, data_dir, phase: str):
        """Initialize this dataset class.

        Args:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            dataset_root datasets/horse2zebra
            max_dataset_size: float(int)
        """
        # create a path '/path/to/data/trainA', '/path/to/data/trainB'
        self.dir_a = os.path.join(data_dir, phase + "A")
        self.dir_b = os.path.join(data_dir, phase + "B")

        self.a_paths = sorted(glob.glob(self.dir_a + "/*.*"))
        self.b_paths = sorted(glob.glob(self.dir_b + "/*.*"))

        self.a_size = len(self.a_paths)  # get the size of dataset A
        self.b_size = len(self.b_paths)  # get the size of dataset B

        self.is_train = phase == "train"

    def __getitem__(self, index: int):
        """Return a data point and its metadata information.

        Args:
            index (int)      -- a random integer for data indexing

        Returns:
            a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        a_path = self.a_paths[index % self.a_size]
        # make sure index is within then range
        # randomize the index for domain B to avoid fixed pairs.
        index_b = random.randint(0, self.b_size - 1)
        b_path = self.b_paths[index_b]
        a_img = Image.open(a_path).convert("RGB")
        b_img = Image.open(b_path).convert("L")
        # apply image transformation
        rgb_transform = get_transform(286, 256, train=self.is_train)
        ir_transform = get_transform(286, 256, grayscale=True, train=self.is_train)
        a_img = rgb_transform(a_img)
        b_img = ir_transform(b_img)

        return a_img, b_img

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.a_size, self.b_size)



class UnalignedRgb2IrDataModule(LightningDataModule):
    """LightningDataModule for the CIFAR10 dataset.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 1,
        num_workers: int = 1,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.trainset: Optional[Dataset] = None
        self.validset: Optional[Dataset] = None
        self.testset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        UnalignedRgb2IrDataset(self.hparams.data_dir, phase="train")

    def setup(self, stage: Optional[str] = None) -> None:
        self.trainset = UnalignedRgb2IrDataset(self.hparams.data_dir, phase="train")
        self.validset = UnalignedRgb2IrDataset(self.hparams.data_dir, phase="test")
        # if not self.trainset and not self.validset and not self.testset:
        #     trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
        #     testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
        #     dataset = ConcatDataset(datasets=[trainset, testset])
        #     self.trainset, self.validset, self.testset = random_split(
        #         dataset=dataset,
        #         lengths=[55_000, 5_000, 10_000],
        #         generator=torch.Generator().manual_seed(42),
        #     )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    # def test_dataloader(self) -> DataLoader[Any]:
    #     return


if __name__ == "__main__":
    _ = UnalignedRgb2IrDataModule()

