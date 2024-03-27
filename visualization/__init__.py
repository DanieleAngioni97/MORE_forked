from torchvision.transforms import Normalize
import numpy as np
import matplotlib.pyplot as plt

class InvNormalize(Normalize):
    def __init__(self, normalizer):
        inv_mean = [-mean / std for mean, std in list(zip(normalizer.mean, normalizer.std))]
        inv_std = [1 / std for std in normalizer.std]
        super().__init__(inv_mean, inv_std)

def _tensor_to_show(img, transforms=None):
    if transforms is not None:
        for transform in transforms.transforms:
            if isinstance(transform, Normalize):
                normalizer = transform
                break
        inverse_transform = InvNormalize(normalizer)
        img = inverse_transform(img)

    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg

def imshow(img, ax=None, transforms=None, figsize=(10, 20), path=None):
    npimg = _tensor_to_show(img, transforms)
    if ax is None:
        plt.figure(figsize=figsize)
        plt.imshow(npimg, interpolation=None)
    else:
        ax.imshow(npimg, interpolation=None)
    if path is not None:
        plt.savefig(path)