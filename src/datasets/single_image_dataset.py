import numpy as np
import os
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from lightning import LightningDataModule
from typing import Optional, Any


def _random_zoom(img, target_width, crop_width, method=Image.BICUBIC, factor=None):
    if factor is None:
        zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    else:
        zoom_level = (factor[0], factor[1])
    iw, ih = img.size
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    img = img.resize((int(round(zoomw)), int(round(zoomh))))
    return img


def _patch(img, index, size):
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    startx = np.random.randint(int(roomx) + 1)
    starty = np.random.randint(int(roomy) + 1)

    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    return img.crop((gridx, gridy, gridx + size, gridy + size))


def get_transform(
    load_size: int = 1024, crop_size: int = 64,
    grayscale: bool = False, scale_factor: float = None, patch_index: int = None
):
    """
    preprocessing = resize_and_crop
    load_size = 286, crop_size=256
    return -1 to 1
    """
    transform_list = []
    # transform_list.append(transforms.Resize(load_size))
    # transform_list.append(
    #     transforms.Lambda(lambda img: _make_power_2(img, base=4, method=method))
    # )
    # if 'zoom' in opt.preprocess:
    transform_list.append(transforms.Lambda(lambda x: _random_zoom(x, load_size, crop_size, scale_factor)))
    # transform_list.append(transforms.RandomCrop(crop_size))
    # if 'patch' in opt.preprocess:
    transform_list.append(transforms.Lambda(lambda x: _patch(x, patch_index, crop_size)))
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor()]
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


class SingleImageDataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    def __init__(self, data_dir, batch_size=16, phase: str = "train", random_scale_max: float = 3.0):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.phase = phase
        self.dir_a = os.path.join(data_dir, phase + "A")
        self.dir_b = os.path.join(data_dir, phase + "B")

        self.A_paths = sorted(glob.glob(self.dir_a + "/*.*"))
        self.B_paths = sorted(glob.glob(self.dir_b + "/*.*"))

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        assert len(self.A_paths) == 1 and len(self.B_paths) == 1,\
            "SingleImageDataset class should be used with one image in each domain"
        A_img = Image.open(self.A_paths[0]).convert('RGB')
        B_img = Image.open(self.B_paths[0]).convert('RGB')
        # print("Image sizes %s and %s" % (str(A_img.size), str(B_img.size)))

        self.A_img = A_img
        self.B_img = B_img

        # In single-image translation, we augment the data loader by applying
        # random scaling. Still, we design the data loader such that the
        # amount of scaling is the same within a minibatch. To do this,
        # we precompute the random scaling values, and repeat them by |batch_size|.
        A_zoom = 1 / random_scale_max
        zoom_levels_A = np.random.uniform(A_zoom, 1.0, size=(len(self) // batch_size + 1, 1, 2))
        self.zoom_levels_A = np.reshape(np.tile(zoom_levels_A, (1, batch_size, 1)), [-1, 2])

        B_zoom = 1 / random_scale_max
        zoom_levels_B = np.random.uniform(B_zoom, 1.0, size=(len(self) // batch_size + 1, 1, 2))
        self.zoom_levels_B = np.reshape(np.tile(zoom_levels_B, (1, batch_size, 1)), [-1, 2])

        # While the crop locations are randomized, the negative samples should
        # not come from the same location. To do this, we precompute the
        # crop locations with no repetition.
        self.patch_indices_A = list(range(len(self)))
        random.shuffle(self.patch_indices_A)
        self.patch_indices_B = list(range(len(self)))
        random.shuffle(self.patch_indices_B)
        # print(self.zoom_levels_A,self.zoom_levels_B)
        # print(self.patch_indices_A,self.patch_indices_B)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # A_path = self.A_paths[0]
        # B_path = self.B_paths[0]
        A_img = self.A_img
        B_img = self.B_img

        # apply image transformation
        if self.phase == "train":
            param = {
                'scale_factor': self.zoom_levels_A[index],
                'patch_index': self.patch_indices_A[index],
                'flip': random.random() > 0.5
            }

            transform_A = get_transform(
                load_size=1024,
                crop_size=64,
                grayscale=False,
                scale_factor=self.zoom_levels_A[index],
                patch_index=self.patch_indices_A[index],
            )
            A = transform_A(A_img)

            transform_B = get_transform(
                load_size=1024,
                crop_size=64,
                grayscale=False,
                scale_factor=self.zoom_levels_B[index],
                patch_index=self.patch_indices_B[index],
            )
            B = transform_B(B_img)
        else:
            transform = get_transform(
                load_size=1024,
                crop_size=64,
                grayscale=False,
                scale_factor=self.zoom_levels_A[index],
                patch_index=self.patch_indices_A[index],
            )
            A = transform(A_img)
            B = transform(B_img)
        return A, B

    def __len__(self):
        """ Let's pretend the single image contains 100,000 crops for convenience.
        """
        return 100000


class SingleImageDataModule(LightningDataModule):
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
        SingleImageDataset(self.hparams.data_dir, self.hparams.batch_size, phase="train")

    def setup(self, stage: Optional[str] = None) -> None:
        self.trainset = SingleImageDataset(self.hparams.data_dir, self.hparams.batch_size, phase="train")

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )


if __name__ == "__main__":
    _ = SingleImageDataModule()

