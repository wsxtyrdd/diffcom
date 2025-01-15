import math
from glob import glob
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets

NUM_DATASET_WORKERS = 6
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class OpenImages(Dataset):
    """OpenImages dataset from [1].

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] https://storage.googleapis.com/openimages/web/factsfigures.html

    """
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, config, data_dir):
        self.imgs = []
        for dir in data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        _, self.im_height, self.im_width = config.img_size
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.scale_min = SCALE_MIN
        self.scale_max = SCALE_MAX
        self.normalize = config.normalize
        self.require_H = False

    def _transforms(self, scale, H, W):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [transforms.ToPILImage(),
                           transforms.RandomHorizontalFlip(),
                           transforms.Resize((math.ceil(scale * H), math.ceil(scale * W))),
                           transforms.RandomCrop((self.im_height, self.im_width))]
        # transforms.Resize((self.im_height, self.im_width)),
        # transforms.ToTensor()]
        transforms_list += [transforms.ToTensor()]
        if self.normalize is True:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        """ TODO: This definitely needs to be optimized.
        Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        """
        # img values already between 0 and 255
        img_path = self.imgs[idx]
        # filesize = os.path.getsize(img_path)
        # This is faster but less convenient
        # H X W X C `ndarray`
        # img = imread(img_path)
        # img_dims = img.shape
        # H, W = img_dims[0], img_dims[1]
        # PIL
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')
        except:
            img_path = self.imgs[idx + 1]
            img = Image.open(img_path)
            img = img.convert('RGB')
            print("ERROR!")
        # except:
        #     img_path = self.imgs[idx + 1]
        #     img = Image.open(img_path)
        #     img = img.convert('RGB')
        #     print("ERROR!")
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img)
        W, H, C = img.shape
        # bpp = filesize * 8. / (H * W)

        shortest_side_length = min(H, W)

        minimum_scale_factor = float(self.crop_size) / float(shortest_side_length)
        scale_low = max(minimum_scale_factor, self.scale_min)
        scale_high = max(scale_low, self.scale_max)
        scale = np.random.uniform(scale_low, scale_high)
        self.dynamic_transform = self._transforms(scale, H, W)
        transformed = self.dynamic_transform(img)
        return transformed

    def __len__(self):
        return len(self.imgs)


def get_cifar10_loader(config):
    dataset_ = datasets.CIFAR10
    dataset_dir = '/home/wangsixian/Dataset/CIFAR10'
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = dataset_(root=dataset_dir,
                             train=True,
                             transform=transform_train,
                             download=False)

    test_dataset = dataset_(root=dataset_dir,
                            train=False,
                            transform=transform_test,
                            download=False)

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=config.batch_size,
                                   num_workers=NUM_DATASET_WORKERS,
                                   drop_last=True,
                                   shuffle=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False)

    return train_loader, test_loader


def get_loader(config):
    train_dataset = OpenImages(config, config.train_data_dir)
    test_dataset = Datasets(config.test_data_dir)

    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=NUM_DATASET_WORKERS,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader


def get_test_loader(config):
    test_dataset = Datasets(config.test_data_dir)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return test_loader


class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        for dir in data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()
        self.normalize = False
        _, self.im_height, self.im_width = (3, 256, 256)
        self.transform = transforms.Compose([  # transforms.RandomHorizontalFlip(0),
            # transforms.CenterCrop((256, 256)),
            transforms.ToTensor()])

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        img = self.transform(image)
        if self.normalize:
            return 2 * img - 1
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.image_path = sorted(glob(data_dir))  # [:100]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)


if __name__ == '__main__':
    import os
    import sys

    sys.path.append("/media/D/wangsixian/DJSCC")
    from ADJSCC.config import config

    config.train_data_dir = ['/home/wangsixian/Dataset/openimages/**']
    dataset = OpenImages(config, config.train_data_dir)
    print(dataset.__len__())
    for i in range(dataset.__len__()):
        # try:
        img = dataset.__getitem__(i)
        print(img.shape)
        break
