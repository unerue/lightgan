import random
from enum import IntEnum
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler


class Initializer(IntEnum):
    NORMAL = 0
    XAVIER_NORMAL = 1
    KAIMING_NORMAL = 2
    ORTHOGONAL = 3


def initialize_weights(m: nn.Module, init_type: Initializer = 1, gain: float = 0.02):
    """
    define the initialization function
    init_type: normal | xavier normal | kaiming normal | orthogonal
    gain `` default to 0.02
    """
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
        if Initializer.NORMAL == init_type:
            nn.init.normal_(m.weight.data, mean=0.0, std=gain)
        elif Initializer.XAVIER_NORMAL == init_type:
            nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif Initializer.KAIMING_NORMAL == init_type:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif Initializer.ORTHOGONAL == init_type:
            nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=gain)
        nn.init.constant_(m.bias.data, 0.0)


def set_requires_grad(nets: nn.Module, requires_grad: bool = False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def custom_schedule(epoch):
    return 1 - max(0, epoch + 1 - 100) / 101


class ImagePool:
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """
    def __init__(self, pool_size: int) -> None:
        """Initialize the ImagePool class

        Args:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs: int = 0
            self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        Return an image from the pool.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.

        Args:
            images: the latest generated images from the generator

        Returns:
            images from the buffer.
        """
        # if the buffer size is 0, do nothing
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            # if the buffer is not full; keep inserting current images to the buffer
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                # by 50% chance, the buffer will return a previously stored image, 
                # and insert the current image into the buffer
                if p > 0.5:
                    # randint is inclusive
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  
                    # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        # collect all the images and return
        return_images = torch.cat(return_images, 0)
        return return_images


class LinearDecayLR(_LRScheduler):
    """
    Custom LR Scheduler which linearly decay to a target learning rate.

    override 해줘야 하는 부분은 get_lr, _get_closed_form_lr
    """
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        target_lr: float,
        total_iters: int,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None:
        if initial_lr < 0:
            raise ValueError("Initial Learning rate expected to be a non-negative integer.")
            
        if target_lr < 0:
            raise ValueError("Target Learning rate expected to be a non-negative integer.")

        if target_lr > initial_lr:
            raise ValueError("Target Learning Rate must be larger than Initial Learning Rate.")

        self.init_lr = initial_lr
        self.target_lr = target_lr
        self.total_iters = total_iters
        self.subtract_lr = self._get_decay_constant()
        super().__init__(optimizer, last_epoch, verbose) # 부모클래스의 init에 필요한 arg 넘겨줌.
                    
    def _get_decay_constant(self):
        return float((self.init_lr - self.target_lr) / self.total_iters)
        
    def get_lr(self):
        # if not self._get_lr_called_within_step:
            # warnings.warn("To get the learning rate computed by the scheduler, "
            #             "please use 'get_last_lr()'.", UserWarning)

        return [group['lr']-self.subtract_lr for group in self.optimizer.param_groups]
    

class DelayedLinearDecayLR(LinearDecayLR):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        target_lr: float,
        total_iters:int,
        last_epoch:int=-1,
        decay_after:int=100,
        verbose:bool=False
    ) -> None:
        self.decay_after = decay_after
        super().__init__(optimizer, initial_lr, target_lr, total_iters, last_epoch, verbose)

    def get_lr(self):
        # if not self._get_lr_called_within_step:
            # warnings.warn("To get the learning rate computed by the scheduler, "
            #             "please use 'get_last_lr()'.", UserWarning)

        # 여기에 total iter도 고려해줘야함.
        if self.decay_after <= self.last_epoch < (self.decay_after + self.total_iters):
            return [group['lr'] - self.subtract_lr for group in self.optimizer.param_groups]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]