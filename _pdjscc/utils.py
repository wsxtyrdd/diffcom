import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import torch
import random
import os
from torch.autograd import Variable
import logging


def CalcuPSNR(img1, img2, max_val=255.):
    """
    Based on `tf.image.psnr`
    https://www.tensorflow.org/api_docs/python/tf/image/psnr
    """
    float_type = 'float64'
    # img1 = (torch.clamp(img1,-1,1).cpu().numpy() + 1) / 2 * 255
    # img2 = (torch.clamp(img2,-1,1).cpu().numpy() + 1) / 2 * 255
    img1 = torch.clamp(img1, 0, 1).cpu().numpy() * 255
    img2 = torch.clamp(img2, 0, 1).cpu().numpy() * 255
    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def MSE2PSNR(MSE):
    return 10 * math.log10(255 ** 2 / (MSE))


def logger_configuration(config, save_log=False, test_mode=False):
    # 配置 logger
    logger = logging.getLogger("Deep joint source channel coder")
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.models)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger


def Var(x, device):
    return Variable(x.to(device))


def single_plot(epoch, global_step, real, gen, config, number, single_compress=False):
    real = real.transpose([1, 2, 0])
    gen = gen.transpose([1, 2, 0])
    images = list()

    for im, imtype in zip([real, gen], ['real', 'gen']):
        # im = ((im + 1.0)) / 2  # [-1,1] -> [0,1]
        im = np.squeeze(im)
        if len(im.shape) == 3:
            im = im[:, :, :3]
        if len(im.shape) == 4:
            im = im[0, :, :, :3]
        images.append(im)

    comparison = np.hstack(images)

    f = plt.figure()
    plt.imshow(comparison)
    plt.axis('off')
    if single_compress:
        f.savefig(config.name, format='png', dpi=720, bbox_inches='tight', pad_inches=0)
    else:
        f.savefig(
            "{}/JSCCModel_{}_epoch{}_step{}_{}.png".format(config.samples, config.trainset, epoch, global_step, number),
            format='png', dpi=720, bbox_inches='tight', pad_inches=0)
    plt.gcf().clear()
    plt.close(f)


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
